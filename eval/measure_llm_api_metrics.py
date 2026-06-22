#!/usr/bin/env python3
"""
MapOMOP LLM API 비용·지연 벤치마크 (논문 보고용).

실제 Stage3/Validation 프롬프트로 API를 호출하고, 토큰·latency·추정 비용을 집계합니다.

Usage:
    # 기존 mapping JSON 로그에서 Stage3만 재실행 (ES 불필요)
    python eval/measure_llm_api_metrics.py --mode replay \\
        --input test_logs/mapping_manual_20260319_132034.json

    # 전체 파이프라인 (ES + SapBERT 필요)
    python eval/measure_llm_api_metrics.py --mode e2e \\
        --input data/manual_test_cases.xlsx --limit 10

    # validation 포함 시 LLM 호출 수 증가 (최대 +3회/엔티티)
    python eval/measure_llm_api_metrics.py --mode replay -i test_logs/foo.json --use-validation

    # 가격표 직접 지정 (USD per 1M tokens)
    python eval/measure_llm_api_metrics.py --mode replay -i ... \\
        --input-price-per-1m 0.25 --output-price-per-1m 2.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

from MapOMOP.entity_mapping_api import DomainID, EntityInput, EntityMappingAPI
from MapOMOP.elasticsearch_client import ElasticsearchClient
from MapOMOP.llm_client import (
    LLMMetricsSummary,
    create_llm_client,
    get_default_price_per_1m,
    summarize_llm_metrics,
)
from MapOMOP.mapping_stages.stage3_hybrid_scoring import Stage3HybridScoring
from MapOMOP.mapping_validation import MappingValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DOMAIN_MAP = {
    "Condition": DomainID.CONDITION,
    "Procedure": DomainID.PROCEDURE,
    "Drug": DomainID.DRUG,
    "Observation": DomainID.OBSERVATION,
    "Measurement": DomainID.MEASUREMENT,
    "Device": DomainID.DEVICE,
}


def _stage2_from_log_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """test_logs JSON의 stage2_candidates → Stage3 입력 형식."""

    candidates = []
    for row in rows:
        concept = {
            "concept_id": row.get("concept_id"),
            "concept_name": row.get("concept_name"),
            "domain_id": row.get("domain_id"),
            "vocabulary_id": row.get("vocabulary_id"),
            "standard_concept": row.get("standard_concept"),
        }
        candidates.append(
            {
                "concept": concept,
                "is_original_standard": row.get("is_original_standard", True),
                "original_candidate": row.get("original_candidate", {}),
                "original_non_standard": row.get("original_non_standard"),
                "relation_type": row.get("relation_type", "original"),
                "elasticsearch_score": row.get("elasticsearch_score", 0.0),
                "search_type": row.get("search_type", "lexical"),
            }
        )
    return candidates


def _load_replay_cases(
    path: Path,
    limit: Optional[int],
    *,
    run_index: int = 0,
) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        if "runs" in raw and raw["runs"]:
            runs = raw["runs"]
            if run_index < 0 or run_index >= len(runs):
                raise ValueError(f"run_index {run_index} out of range (0..{len(runs) - 1})")
            cases = runs[run_index]
        else:
            cases = raw.get("results") or raw.get("cases") or [raw]
    else:
        cases = raw

    filtered = [c for c in cases if c.get("stage2_candidates")]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def _aggregate_run_summaries(
    run_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """여러 반복 실행의 paper_snippet/latency/cost 평균."""

    if not run_summaries:
        return {}

    def _mean(key_path: str) -> float:
        values = []
        for run in run_summaries:
            node = run
            for key in key_path.split("."):
                node = node[key]
            values.append(float(node))
        return sum(values) / len(values)

    return {
        "repeat_count": len(run_summaries),
        "mean_latency_ms_per_call": round(_mean("paper_snippet.mean_latency_ms"), 2),
        "mean_p95_latency_ms_per_call": round(_mean("paper_snippet.p95_latency_ms"), 2),
        "mean_tokens_per_entity": round(_mean("paper_snippet.tokens_per_entity"), 1),
        "mean_cost_usd_per_entity": round(_mean("paper_snippet.cost_usd_per_entity"), 6),
        "mean_cost_usd_total_per_run": round(_mean("cost_usd.total"), 4),
        "mean_llm_calls_per_run": round(_mean("llm_calls.total"), 1),
    }


def run_replay_benchmark(
    cases: List[Dict[str, Any]],
    *,
    scoring_mode: str,
    use_validation: bool,
    llm_client,
) -> LLMMetricsSummary:
    stage3 = Stage3HybridScoring(
        scoring_mode=scoring_mode,
        llm_client=llm_client,
    )
    validator = None
    if use_validation:
        validator = MappingValidator(llm_client=llm_client)

    for idx, case in enumerate(cases, 1):
        entity_name = str(case.get("entity_name", "")).strip()
        stage2 = _stage2_from_log_rows(case["stage2_candidates"])
        stage3.calculate_hybrid_scores(
            entity_name=entity_name,
            stage2_candidates=stage2,
            stage1_candidates=[],
            entity_embedding=None,
        )

        if validator and case.get("stage3_candidates"):
            top = case["stage3_candidates"][0]
            validator.validate_mapping(
                entity_name=entity_name,
                concept_id=str(top.get("concept_id", "")),
                concept_name=str(top.get("concept_name", "")),
                synonyms=[],
            )

        if idx % 10 == 0:
            logger.info("replay 진행: %d/%d", idx, len(cases))

    return llm_client.summarize_metrics()


def _load_e2e_rows(path: Path, limit: Optional[int]) -> List[Dict[str, str]]:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        import pandas as pd

        df = pd.read_excel(path)
        rows = df.to_dict("records")
    else:
        import pandas as pd

        df = pd.read_csv(path)
        rows = df.to_dict("records")

    if limit is not None:
        rows = rows[:limit]
    return rows


def run_e2e_benchmark(
    rows: List[Dict[str, str]],
    *,
    scoring_mode: str,
    use_validation: bool,
    llm_client,
) -> LLMMetricsSummary:
    es_client = ElasticsearchClient()
    api = EntityMappingAPI(
        es_client=es_client,
        scoring_mode=scoring_mode,
        use_validation=use_validation,
        llm_client=llm_client,
    )

    for idx, row in enumerate(rows, 1):
        entity_name = str(
            row.get("entity_name") or row.get("entity") or row.get("source_value") or ""
        ).strip()
        domain_raw = str(row.get("domain_id") or row.get("domain") or row.get("input_domain") or "")
        domain_id = DOMAIN_MAP.get(domain_raw)
        if not entity_name or domain_id is None:
            logger.warning("스킵 (entity/domain 없음): %s", row)
            continue

        entity = EntityInput(entity_name=entity_name, domain_id=domain_id)
        api.map_entity(entity)

        if idx % 10 == 0:
            logger.info("e2e 진행: %d/%d", idx, len(rows))

    return llm_client.summarize_metrics()


def _summary_to_report(
    summary: LLMMetricsSummary,
    *,
    mode: str,
    entity_count: int,
    scoring_mode: str,
    use_validation: bool,
    model: str,
    provider: str,
    input_price_per_1m: float,
    output_price_per_1m: float,
) -> Dict[str, Any]:
    per_entity_calls = summary.call_count / entity_count if entity_count else 0.0
    per_entity_cost = summary.total_cost_usd / entity_count if entity_count else 0.0
    per_entity_latency = summary.mean_latency_ms * per_entity_calls if entity_count else 0.0

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "entity_count": entity_count,
        "scoring_mode": scoring_mode,
        "use_validation": use_validation,
        "llm": {"provider": provider, "model": model},
        "pricing_usd_per_1m": {
            "input": input_price_per_1m,
            "output": output_price_per_1m,
        },
        "llm_calls": {
            "total": summary.call_count,
            "success": summary.success_count,
            "per_entity_mean": round(per_entity_calls, 3),
            "by_tag": summary.by_tag,
        },
        "latency_ms": {
            "mean_per_call": round(summary.mean_latency_ms, 2),
            "p50_per_call": round(summary.p50_latency_ms, 2),
            "p95_per_call": round(summary.p95_latency_ms, 2),
            "mean_per_entity_estimated": round(per_entity_latency, 2),
        },
        "tokens": {
            "total_input": summary.total_input_tokens,
            "total_output": summary.total_output_tokens,
            "total": summary.total_tokens,
            "mean_per_call": round(
                summary.total_tokens / summary.success_count, 1
            )
            if summary.success_count
            else 0,
            "mean_per_entity": round(
                summary.total_tokens / entity_count, 1
            )
            if entity_count
            else 0,
        },
        "cost_usd": {
            "total": round(summary.total_cost_usd, 6),
            "mean_per_call": round(summary.mean_cost_usd_per_call, 6),
            "mean_per_entity": round(per_entity_cost, 6),
        },
        "paper_snippet": {
            "mean_latency_ms": round(summary.mean_latency_ms, 1),
            "p95_latency_ms": round(summary.p95_latency_ms, 1),
            "cost_usd_per_entity": round(per_entity_cost, 5),
            "tokens_per_entity": round(
                summary.total_tokens / entity_count, 0
            )
            if entity_count
            else 0,
        },
    }


def _print_report(report: Dict[str, Any]) -> None:
    llm = report["llm"]
    lat = report["latency_ms"]
    tok = report["tokens"]
    cost = report["cost_usd"]
    paper = report["paper_snippet"]

    print("\n=== MapOMOP LLM API 벤치마크 ===")
    print(f"mode={report['mode']} entities={report['entity_count']} "
          f"scoring={report['scoring_mode']} validation={report['use_validation']}")
    print(f"model={llm['model']} provider={llm['provider']}")
    print(f"LLM calls: {report['llm_calls']['total']} "
          f"(≈{report['llm_calls']['per_entity_mean']}/entity)")
    print(f"Latency (ms/call): mean={lat['mean_per_call']} "
          f"p50={lat['p50_per_call']} p95={lat['p95_per_call']}")
    print(f"Latency (ms/entity est.): {lat['mean_per_entity_estimated']}")
    print(f"Tokens: total={tok['total']} per_entity≈{tok['mean_per_entity']}")
    print(f"Cost (USD): total={cost['total']} per_entity≈{cost['mean_per_entity']}")
    print("\n--- 논문용 한 줄 요약 (마지막 반복) ---")
    print(
        f"Mean LLM latency {paper['mean_latency_ms']} ms/call "
        f"(p95 {paper['p95_latency_ms']} ms); "
        f"≈{paper['tokens_per_entity']} tokens and "
        f"${paper['cost_usd_per_entity']} per entity "
        f"({llm['model']})."
    )
    agg = report.get("repeats_aggregate")
    if agg and agg.get("repeat_count", 1) > 1:
        print("\n--- {}회 반복 평균 ---".format(agg["repeat_count"]))
        print(
            f"Mean latency {agg['mean_latency_ms_per_call']} ms/call "
            f"(p95 {agg['mean_p95_latency_ms_per_call']} ms); "
            f"≈{agg['mean_tokens_per_entity']} tokens/entity; "
            f"${agg['mean_cost_usd_per_entity']}/entity; "
            f"${agg['mean_cost_usd_total_per_run']}/run"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="MapOMOP LLM cost/latency benchmark")
    parser.add_argument(
        "--mode",
        choices=["replay", "e2e"],
        default="replay",
        help="replay: mapping JSON의 stage2로 Stage3만; e2e: 전체 파이프라인",
    )
    parser.add_argument("--input", "-i", required=True, help="JSON(replay) 또는 CSV/XLSX(e2e)")
    parser.add_argument("--output", "-o", default=None, help="JSON 리포트 경로 (기본: eval/out/...)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scoring-mode", default="llm", choices=["llm", "llm_with_score"])
    parser.add_argument("--use-validation", action="store_true")
    parser.add_argument("--input-price-per-1m", type=float, default=None)
    parser.add_argument("--output-price-per-1m", type=float, default=None)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="동일 데이터셋 반복 측정 횟수 (예: 5)",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=0,
        help="num_runs/runs 형식 JSON에서 사용할 run 인덱스",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("입력 파일 없음: %s", input_path)
        return 1

    llm_client = create_llm_client(enable_metrics=True)
    if not llm_client.is_initialized:
        logger.error("LLM 클라이언트 초기화 실패. API 키 및 .env를 확인하세요.")
        return 1

    info = llm_client.get_info()
    default_in, default_out = get_default_price_per_1m(info["provider"], info["model"])
    input_price = args.input_price_per_1m if args.input_price_per_1m is not None else default_in
    output_price = args.output_price_per_1m if args.output_price_per_1m is not None else default_out
    llm_client.input_price_per_1m = input_price
    llm_client.output_price_per_1m = output_price

    repeats = max(1, args.repeats)
    run_reports: List[Dict[str, Any]] = []
    entity_count = 0

    if args.mode == "replay":
        cases = _load_replay_cases(
            input_path, args.limit, run_index=args.run_index
        )
        if not cases:
            logger.error("replay 가능한 case(stage2_candidates)가 없습니다.")
            return 1
        entity_count = len(cases)
        benchmark_rows = None
    else:
        benchmark_rows = _load_e2e_rows(input_path, args.limit)
        entity_count = len(benchmark_rows)

    for repeat_idx in range(1, repeats + 1):
        llm_client.reset_metrics()
        logger.info("=== 반복 %d/%d 시작 ===", repeat_idx, repeats)

        if args.mode == "replay":
            summary = run_replay_benchmark(
                cases,
                scoring_mode=args.scoring_mode,
                use_validation=args.use_validation,
                llm_client=llm_client,
            )
        else:
            summary = run_e2e_benchmark(
                benchmark_rows,
                scoring_mode=args.scoring_mode,
                use_validation=args.use_validation,
                llm_client=llm_client,
            )

        run_report = _summary_to_report(
            summary,
            mode=args.mode,
            entity_count=entity_count,
            scoring_mode=args.scoring_mode,
            use_validation=args.use_validation,
            model=info["model"],
            provider=info["provider"],
            input_price_per_1m=input_price,
            output_price_per_1m=output_price,
        )
        run_report["repeat_index"] = repeat_idx
        run_reports.append(run_report)
        logger.info(
            "반복 %d 완료: mean_latency=%.1f ms, cost/run=$%.4f",
            repeat_idx,
            run_report["paper_snippet"]["mean_latency_ms"],
            run_report["cost_usd"]["total"],
        )

    report = run_reports[-1]
    report["repeats"] = repeats
    report["runs"] = run_reports
    report["repeats_aggregate"] = _aggregate_run_summaries(run_reports)
    report["raw_records"] = [asdict(r) for r in llm_client.get_metrics_records()]

    out_dir = _root / "eval" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"llm_api_metrics_{args.mode}_{stamp}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("리포트 저장: %s", out_path)

    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
