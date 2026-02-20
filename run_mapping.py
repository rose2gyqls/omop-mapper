#!/usr/bin/env python3
"""
통합 매핑 실행 스크립트

데이터 소스(snuh, snomed 등) 선택 시 기본 CSV 경로 및 전처리 적용.
동일한 timestamp로 JSON(raw), LOG, XLSX 3개 파일을 test_logs/에 생성합니다.

Usage:
    python run_mapping.py snuh
    python run_mapping.py snomed
    python run_mapping.py snuh --sample-per-domain 5 --random
"""

import argparse
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

from MapOMOP.entity_mapping_api import EntityMappingAPI, EntityInput, DomainID
from MapOMOP.elasticsearch_client import ElasticsearchClient

from mapping_common import (
    DATA_SOURCES,
    load_snuh_data,
    load_snomed_data,
    snuh_row_to_input,
    snomed_row_to_input,
    setup_logging,
    save_json,
    save_xlsx,
)

DOMAIN_MAP = {
    "Condition": DomainID.CONDITION,
    "Procedure": DomainID.PROCEDURE,
    "Drug": DomainID.DRUG,
    "Observation": DomainID.OBSERVATION,
    "Measurement": DomainID.MEASUREMENT,
    "Period": DomainID.PERIOD,
    "Provider": DomainID.PROVIDER,
    "Device": DomainID.DEVICE,
}


def run_mapping(
    data_type: str,
    output_dir: str = "test_logs",
    sample_size: int = 10000,
    use_random: bool = False,
    random_state: int = 42,
    sample_per_domain: int | None = None,
    scoring_mode: str = "llm",
):
    """매핑 실행: 데이터 로드(기본 경로+전처리) → 매핑 → JSON/LOG/XLSX 출력."""
    from datetime import datetime
    from tqdm import tqdm

    if data_type not in DATA_SOURCES:
        raise ValueError(f"Unknown data_type: {data_type}. Available: {list(DATA_SOURCES.keys())}")

    config = DATA_SOURCES[data_type]
    csv_path = config["csv_path"]
    id_col = config["id_col"]

    # sample_per_domain: 정수 N → 도메인별 N개
    sample_per_domain_dict = None
    if sample_per_domain is not None:
        sample_per_domain_dict = {d: sample_per_domain for d in config["domains"]}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger, _ = setup_logging(out_path, data_type, timestamp)
    logger.info("=" * 80)
    logger.info(f"매핑 시작: data={data_type}, csv={csv_path}")
    logger.info(f"전처리: {config.get('vocabulary_filter', config.get('filter_domains', '없음'))}")
    logger.info(f"output_dir={out_path}")
    logger.info("=" * 80)

    # 데이터 로드 (전처리 적용)
    if data_type == "snuh":
        df = load_snuh_data(
            csv_path,
            sample_size=sample_size,
            use_random=use_random,
            random_state=random_state,
            sample_per_domain=sample_per_domain_dict,
            vocabulary_filter=config.get("vocabulary_filter"),
        )
        row_to_input = snuh_row_to_input
    else:
        df = load_snomed_data(
            csv_path,
            sample_size=sample_size,
            use_random=use_random,
            random_state=random_state,
            sample_per_domain=sample_per_domain_dict,
            filter_domains=config.get("filter_domains"),
        )
        row_to_input = snomed_row_to_input

    logger.info(f"로드된 데이터: {len(df)}행")

    es_client = ElasticsearchClient()
    api = EntityMappingAPI(es_client=es_client, scoring_mode=scoring_mode)
    logger.info(f"Scoring mode: {scoring_mode}")

    results = []
    start_time = time.time()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="매핑"):
        try:
            entity_name, domain_id, record_id, ground_truth = row_to_input(row, DOMAIN_MAP)
            entity_input = EntityInput(
                entity_name=entity_name,
                domain_id=domain_id,
                vocabulary_id=None,
            )

            mapping_results = api.map_entity(entity_input)

            stage1 = getattr(api, "_last_stage1_candidates", []) or []
            stage2 = getattr(api, "_last_stage2_candidates", []) or []
            stage3 = getattr(api, "_last_rerank_candidates", []) or []

            best = None
            if mapping_results:
                best = max(mapping_results, key=lambda x: x.mapping_score)

            mapping_correct = False
            if best and ground_truth is not None:
                try:
                    mapping_correct = int(best.mapped_concept_id) == int(ground_truth)
                except (ValueError, TypeError):
                    pass

            r = {
                "test_index": idx + 1,
                "id": record_id,
                id_col: record_id,
                "entity_name": entity_name,
                "input_domain": domain_id.value if domain_id else "All",
                "ground_truth_concept_id": ground_truth,
                "success": mapping_results is not None and len(mapping_results) > 0,
                "mapping_correct": mapping_correct,
                "best_result_domain": best.domain_id if best else None,
                "best_concept_id": best.mapped_concept_id if best else None,
                "best_concept_name": best.mapped_concept_name if best else None,
                "best_score": best.mapping_score if best else 0.0,
                "stage1_candidates": stage1,
                "stage2_candidates": stage2,
                "stage3_candidates": stage3,
            }
            results.append(r)

            if idx < 5:
                status = "정답" if mapping_correct else ("오답" if r["success"] else "실패")
                logger.info(f"#{idx + 1} {entity_name}: {status}")

        except Exception as e:
            logger.error(f"#{idx + 1} 오류: {e}")
            entity_col = "source_name" if data_type == "snuh" else "entity_name"
            rid = str(row.get(id_col, "N/A"))
            results.append({
                "test_index": idx + 1,
                "id": rid,
                id_col: rid,
                "entity_name": str(row.get(entity_col, "")),
                "input_domain": "All",
                "ground_truth_concept_id": None,
                "success": False,
                "mapping_correct": False,
                "best_result_domain": None,
                "best_concept_id": None,
                "best_concept_name": None,
                "best_score": 0.0,
                "stage1_candidates": [],
                "stage2_candidates": [],
                "stage3_candidates": [],
                "error": str(e),
            })

    elapsed = time.time() - start_time

    total = len(results)
    success_count = sum(1 for r in results if r["success"])
    correct_count = sum(1 for r in results if r["mapping_correct"])

    logger.info("")
    logger.info("=" * 80)
    logger.info("결과 요약")
    logger.info("=" * 80)
    logger.info(f"총: {total}개, 성공: {success_count}개 ({100 * success_count / total:.2f}%), 정답: {correct_count}개 ({100 * correct_count / total:.2f}%)")
    logger.info(f"소요: {elapsed:.2f}초 ({elapsed / 60:.2f}분)")

    json_path = save_json(results, out_path, data_type, timestamp)
    xlsx_path = save_xlsx(results, out_path, data_type, timestamp)
    logger.info(f"JSON: {json_path}")
    logger.info(f"XLSX: {xlsx_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="OMOP 매핑 통합 실행")
    parser.add_argument(
        "data",
        choices=list(DATA_SOURCES.keys()),
        help=f"데이터 소스 (기본 CSV 및 전처리 자동 적용). 사용 가능: {list(DATA_SOURCES.keys())}",
    )
    parser.add_argument("--sample-size", "-n", type=int, default=10000, help="샘플 크기 (sample-per-domain 미사용 시)")
    parser.add_argument("--random", action="store_true", help="랜덤 샘플링")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument(
        "--sample-per-domain",
        type=int,
        default=None,
        metavar="N",
        help="도메인별 N개씩 샘플. 예: --sample-per-domain 5",
    )
    parser.add_argument(
        "--scoring",
        default="llm",
        choices=["llm", "llm_with_score", "semantic", "hybrid"],
        help="Scoring mode (ablation study용)",
    )

    args = parser.parse_args()

    run_mapping(
        data_type=args.data,
        output_dir="test_logs",
        sample_size=args.sample_size,
        use_random=args.random,
        random_state=args.seed,
        sample_per_domain=args.sample_per_domain,
        scoring_mode=args.scoring,
    )


if __name__ == "__main__":
    main()
