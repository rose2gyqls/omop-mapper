#!/usr/bin/env python3
"""
통합 매핑 실행 스크립트

데이터 소스(snuh, snomed 등) 선택 시 기본 CSV 경로 및 전처리 적용.
동일한 timestamp로 JSON(raw), LOG, XLSX 3개 파일을 test_logs/에 생성합니다.

Usage:
    python run_mapping.py snuh
    python run_mapping.py snomed
    python run_mapping.py snuh --sample-per-domain 5 --random
    python run_mapping.py snuh --workers 4   # 병렬 처리
"""

import argparse
import functools
import os
import sys
import time

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

from MapOMOP.entity_mapping_api import EntityMappingAPI, EntityInput, DomainID
from MapOMOP.elasticsearch_client import ElasticsearchClient

from mapping_common import (
    API_LOGGER_NAMES,
    DATA_SOURCES,
    load_snuh_data,
    load_snomed_data,
    snuh_row_to_input,
    snomed_row_to_input,
    setup_logging,
    setup_worker_logging,
    capture_entity_logs,
    save_json,
    save_xlsx,
    save_xlsx_repeat,
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

# ProcessPoolExecutor worker용 전역 (각 프로세스에서 초기화)
_worker_api = None


def _worker_init(
    scoring_mode: str,
    use_validation: bool = False,
    log_file_path: str | None = None,
    capture_only: bool = False,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    llm_temperature: float | None = None,
    llm_top_p: float | None = None,
    llm_max_tokens: int | None = None,
):
    """Worker 프로세스 초기화: API 인스턴스 1회 생성 + 로깅 설정.
    capture_only=True: 로그를 캡처 모드로 (멀티 워커 시 엔티티별로 묶어서 출력).
    """
    global _worker_api
    if log_file_path:
        setup_worker_logging(log_file_path, capture_only=capture_only)
    es_client = ElasticsearchClient()
    _worker_api = EntityMappingAPI(
        es_client=es_client,
        scoring_mode=scoring_mode,
        use_validation=use_validation,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_temperature=llm_temperature,
        llm_top_p=llm_top_p,
        llm_max_tokens=llm_max_tokens,
    )


def _map_single_task(task, capture_logs: bool = False):
    """단일 엔티티 매핑 (worker에서 실행, pickle 가능한 인자만 사용).
    capture_logs=True: API 로그를 캡처해 (test_index, result, log_lines) 반환.
    task: (row_idx, test_index, entity_name, domain_str, record_id, ground_truth, ground_truth_concept_name, id_col)
    """
    global _worker_api
    (row_idx, test_index, entity_name, domain_str, record_id, ground_truth, ground_truth_concept_name, id_col) = task
    domain_id = DOMAIN_MAP.get(domain_str) if domain_str else None
    entity_input = EntityInput(
        entity_name=entity_name,
        domain_id=domain_id,
        vocabulary_id=None,
    )

    handlers_added = []
    log_lines: list[str] = []
    if capture_logs:
        handlers_added, log_lines = capture_entity_logs(API_LOGGER_NAMES)

    try:
        mapping_results = _worker_api.map_entity(entity_input)
        stage1 = getattr(_worker_api, "_last_stage1_candidates", []) or []
        stage2 = getattr(_worker_api, "_last_stage2_candidates", []) or []
        stage3 = getattr(_worker_api, "_last_rerank_candidates", []) or []

        best = None
        if mapping_results:
            best = max(mapping_results, key=lambda x: x.mapping_score)

        mapping_correct = False
        if best and ground_truth is not None:
            try:
                mapping_correct = int(best.mapped_concept_id) == int(ground_truth)
            except (ValueError, TypeError):
                pass

        result = {
                "test_index": test_index,
                "id": record_id,
                "entity_name": entity_name,
                "input_domain": domain_id.value if domain_id else "All",
        }
        if id_col != "test_index":
            result[id_col] = record_id
        result.update({
            "ground_truth_concept_id": ground_truth,
            "ground_truth_concept_name": ground_truth_concept_name,
            "success": mapping_results is not None and len(mapping_results) > 0,
            "mapping_correct": mapping_correct,
            "best_result_domain": best.domain_id if best else None,
            "best_concept_id": best.mapped_concept_id if best else None,
            "best_concept_name": best.mapped_concept_name if best else None,
            "best_score": best.mapping_score if best else 0.0,
            "stage1_candidates": stage1,
            "stage2_candidates": stage2,
            "stage3_candidates": stage3,
        })
        if capture_logs:
            h = handlers_added[0][1]
            for log, _ in handlers_added:
                log.removeHandler(h)
        if capture_logs:
            return (test_index, result, log_lines)
        return (test_index, result)
    except Exception as e:
        if capture_logs:
            h = handlers_added[0][1]
            for log, _ in handlers_added:
                log.removeHandler(h)
        result = {
            "test_index": test_index,
            "id": record_id,
            "entity_name": entity_name,
            "input_domain": "All",
            "ground_truth_concept_id": ground_truth,
            "ground_truth_concept_name": ground_truth_concept_name,
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
        }
        if id_col != "test_index":
            result[id_col] = record_id
        if capture_logs:
            return (test_index, result, log_lines)
        return (test_index, result)


def run_mapping(
    data_type: str,
    output_dir: str = "test_logs",
    sample_size: int | None = None,
    use_random: bool = False,
    random_state: int = 42,
    sample_per_domain: int | None = None,
    scoring_mode: str = "llm",
    workers: int = 1,
    num_runs: int = 1,
    use_validation: bool = False,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    llm_temperature: float | None = None,
    llm_top_p: float | None = None,
    llm_max_tokens: int | None = None,
):
    """매핑 실행: 데이터 로드(기본 경로+전처리) → 매핑 → JSON/LOG/XLSX 출력.
    workers > 1 이면 ProcessPoolExecutor로 병렬 처리.
    num_runs > 1 이면 동일 데이터로 N회 반복 (일관성 검증용).
    use_validation=False(기본): stage 1~3 점수 기반 최고 점수 매핑만 사용.
    use_validation=True: LLM validation 모듈 포함. 출력 파일명에 _withval 붙음.
    """
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

    data_type_out = f"{data_type}_withval" if use_validation else data_type
    # workers > 1: 터미널에는 tqdm 진행률만, 상세로그는 파일로
    logger, log_file = setup_logging(out_path, data_type_out, timestamp, console=(workers == 1))
    logger.info("=" * 80)
    logger.info(f"매핑 시작: data={data_type}, csv={csv_path}")
    logger.info(f"전처리: {config.get('vocabulary_filter', config.get('filter_domains', '없음'))}")
    logger.info(f"output_dir={out_path}")
    if llm_provider or llm_model or llm_base_url or llm_temperature is not None or llm_top_p is not None or llm_max_tokens is not None:
        logger.info(
            "LLM override: provider=%s, model=%s, base_url=%s, temperature=%s, top_p=%s, max_tokens=%s",
            llm_provider or "(env/default)",
            llm_model or "(env/default)",
            llm_base_url or "(env/default)",
            llm_temperature if llm_temperature is not None else "(env/default)",
            llm_top_p if llm_top_p is not None else "(env/default)",
            llm_max_tokens if llm_max_tokens is not None else "(env/default)",
        )
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
            filter_domains=config.get("filter_domains"),
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

    workers = max(1, int(workers))
    num_runs = max(1, int(num_runs))
    logger.info(f"로드된 데이터: {len(df)}행")
    logger.info(f"Scoring mode: {scoring_mode}, Workers: {workers}, Runs: {num_runs}, Validation: {'on' if use_validation else 'off'}")

    all_results = []  # num_runs > 1 일 때 [run1_results, run2_results, ...]
    start_time = time.time()

    for run_idx in range(num_runs):
        if num_runs > 1:
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"매핑 Run {run_idx + 1}/{num_runs}")
            logger.info("=" * 80)

        results = []

        # 병렬: ProcessPoolExecutor (workers > 1)
        if workers > 1:
            tasks = []
            for idx, row in df.iterrows():
                entity_name, domain_id, record_id, ground_truth, ground_truth_concept_name = row_to_input(row, DOMAIN_MAP)
                domain_str = domain_id.value if domain_id else None
                # SNOMED: CSV의 test_index 사용, 그 외: 행 인덱스+1
                csv_test_index = int(row["test_index"]) if "test_index" in df.columns and pd.notna(row.get("test_index")) else (idx + 1)
                tasks.append((idx, csv_test_index, entity_name, domain_str, record_id, ground_truth, ground_truth_concept_name, id_col))

            completed = 0
            log_buffer = {}  # row_idx -> log_lines (완료 순서가 아닌 엔티티 순서로 저장)
            map_task = functools.partial(_map_single_task, capture_logs=True)
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_worker_init,
                initargs=(
                    scoring_mode,
                    use_validation,
                    str(log_file),
                    True,
                    llm_provider,
                    llm_model,
                    llm_base_url,
                    llm_api_key,
                    llm_temperature,
                    llm_top_p,
                    llm_max_tokens,
                ),
            ) as ex:
                future_to_row_idx = {ex.submit(map_task, t): t[0] for t in tasks}
                indexed_results = [None] * len(tasks)
                with tqdm(total=len(tasks), desc="매핑", unit="건") as pbar:
                    for future in as_completed(future_to_row_idx):
                        row_idx = future_to_row_idx[future]
                        try:
                            ret = future.result()
                            if len(ret) == 3:
                                _, r, log_lines = ret
                                if log_lines:
                                    log_buffer[row_idx] = log_lines
                            else:
                                _, r = ret
                            indexed_results[row_idx] = r
                            completed += 1
                            if completed <= 5:
                                status = "정답" if r.get("mapping_correct") else ("오답" if r.get("success") else "실패")
                                logger.info(f"#{r.get('test_index')} {r['entity_name']}: {status}")
                            pbar.update(1)
                        except Exception as e:
                            task = tasks[row_idx]
                            test_index = task[1]
                            logger.error(f"#{test_index} Worker 예외: {e}")
                            indexed_results[row_idx] = {
                                "test_index": test_index,
                                "id": task[4],
                                "entity_name": task[2],
                                "input_domain": task[3] or "All",
                                "ground_truth_concept_id": task[5],
                                "ground_truth_concept_name": task[6],
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
                            }
                            pbar.update(1)
                # 로그: 엔티티 순서(row_idx)대로 파일에 기록
                if log_buffer:
                    with open(log_file, "a", encoding="utf-8") as f:
                        for row_idx in range(len(tasks)):
                            if row_idx in log_buffer:
                                r = indexed_results[row_idx]
                                if r:
                                    f.write(f"\n{'='*60} Entity #{r.get('test_index')}: {r.get('entity_name', 'N/A')} {'='*60}\n")
                                for line in log_buffer[row_idx]:
                                    f.write(line + "\n")
            results = [r for r in indexed_results if r is not None]
        else:
            # 순차 처리 (workers == 1)
            es_client = ElasticsearchClient()
            api = EntityMappingAPI(
                es_client=es_client,
                scoring_mode=scoring_mode,
                use_validation=use_validation,
                llm_provider=llm_provider,
                llm_model=llm_model,
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key,
                llm_temperature=llm_temperature,
                llm_top_p=llm_top_p,
                llm_max_tokens=llm_max_tokens,
            )
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="매핑"):
                try:
                    entity_name, domain_id, record_id, ground_truth, ground_truth_concept_name = row_to_input(row, DOMAIN_MAP)
                    # SNOMED: CSV의 test_index 사용, 그 외: 행 인덱스+1
                    csv_test_index = int(row["test_index"]) if "test_index" in df.columns and pd.notna(row.get("test_index")) else (idx + 1)

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
                        "test_index": csv_test_index,
                        "id": record_id,
                        "entity_name": entity_name,
                        "input_domain": domain_id.value if domain_id else "All",
                        "ground_truth_concept_id": ground_truth,
                        "ground_truth_concept_name": ground_truth_concept_name,
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
                        logger.info(f"#{csv_test_index} {entity_name}: {status}")

                except Exception as e:
                    csv_test_index = int(row["test_index"]) if "test_index" in df.columns and pd.notna(row.get("test_index")) else (idx + 1)
                    logger.error(f"#{csv_test_index} 오류: {e}")
                    entity_col = "source_name" if data_type == "snuh" else "entity_name"
                    rid = str(row.get("row_id", row.get("note_id", "N/A")))
                    results.append({
                        "test_index": csv_test_index,
                        "id": rid,
                        "entity_name": str(row.get(entity_col, "")),
                        "input_domain": "All",
                        "ground_truth_concept_id": None,
                        "ground_truth_concept_name": None,
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

        all_results.append(results)

        # num_runs > 1: 각 run 완료 시마다 JSON/XLSX 즉시 저장 (전체 완료 대기 불필요)
        if num_runs > 1:
            completed_runs = len(all_results)
            save_json({"num_runs": completed_runs, "runs": all_results}, out_path, data_type_out, timestamp)
            save_xlsx_repeat(all_results, out_path, data_type_out, timestamp)
            logger.info(f"Run {completed_runs}/{num_runs} 완료 → JSON/XLSX 저장됨")

    elapsed = time.time() - start_time

    # 요약: 마지막 run 기준 (단일 run과 동일 포맷)
    results = all_results[-1]
    total = len(results)
    success_count = sum(1 for r in results if r["success"])
    correct_count = sum(1 for r in results if r["mapping_correct"])

    logger.info("")
    logger.info("=" * 80)
    logger.info("결과 요약")
    logger.info("=" * 80)
    logger.info(f"총: {total}개, 성공: {success_count}개 ({100 * success_count / total:.2f}%), 정답: {correct_count}개 ({100 * correct_count / total:.2f}%)")
    logger.info(f"소요: {elapsed:.2f}초 ({elapsed / 60:.2f}분)")

    if workers > 1:
        print(f"\n총 {total}건, 성공 {success_count}건 ({100 * success_count / total:.1f}%), 정답 {correct_count}건 | 소요 {elapsed / 60:.1f}분")

    if num_runs > 1:
        all_same_count = 0
        for row_idx in range(total):
            concept_ids = [
                str(all_results[run][row_idx].get("best_concept_id") or "")
                for run in range(num_runs)
            ]
            if len(set(concept_ids)) == 1:
                all_same_count += 1
        logger.info(f"{num_runs}회 동일 결과: {all_same_count}/{total}개 ({100 * all_same_count / total:.2f}%)")
        # 이미 각 run 완료 시 저장됨. 최종 경로만 로깅
        json_path = out_path / f"mapping_{data_type_out}_{timestamp}.json"
        xlsx_path = out_path / f"mapping_{data_type_out}_{timestamp}.xlsx"
    else:
        json_path = save_json(results, out_path, data_type_out, timestamp)
        xlsx_path = save_xlsx(results, out_path, data_type_out, timestamp)
    logger.info(f"JSON: {json_path}")
    logger.info(f"XLSX: {xlsx_path}")

    return all_results if num_runs > 1 else results


def main():
    parser = argparse.ArgumentParser(description="OMOP 매핑 통합 실행")
    parser.add_argument(
        "data",
        choices=list(DATA_SOURCES.keys()),
        help=f"데이터 소스 (기본 CSV 및 전처리 자동 적용). 사용 가능: {list(DATA_SOURCES.keys())}",
    )
    parser.add_argument("--sample-size", "-n", type=int, default=None, help="샘플 크기 (-n 미지정 시 전체 데이터, sample-per-domain 미사용 시 적용)")
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
        choices=["llm", "llm_with_score", "semantic"],
        help="Scoring mode (ablation study용)",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "together"],
        default=None,
        help="LLM route 선택. 미지정 시 LLM_PROVIDER/env 기본값 사용.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="모델명 override. Together는 gpt_oss_20b, mistral_small_24b, llama4_maverick alias도 지원.",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="OpenAI-compatible base URL override.",
    )
    parser.add_argument(
        "--llm-api-key-env",
        default=None,
        help="API key를 읽을 환경변수 이름. 예: OPENAI_API_KEY",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=None,
        help="LLM temperature override.",
    )
    parser.add_argument(
        "--llm-top-p",
        type=float,
        default=None,
        help="LLM top_p override.",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=None,
        help="LLM max output tokens override. 미지정 시 env/default 사용.",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        metavar="N",
        help="병렬 워커 수 (기본: 1, 순차 처리). 4~8 권장.",
    )
    parser.add_argument(
        "--repeat",
        "-r",
        type=int,
        default=1,
        metavar="N",
        help="동일 데이터로 N회 매핑 반복 (일관성 검증용). 5 입력 시 현황+5개 상세 시트 생성.",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="LLM validation 모듈 포함 (기본: stage 1~3 점수 기반 최고 점수만 사용). 출력: mapping_{snuh|snomed}_withval_{timestamp}.*",
    )

    args = parser.parse_args()
    llm_api_key = os.getenv(args.llm_api_key_env) if args.llm_api_key_env else None

    run_mapping(
        data_type=args.data,
        output_dir="test_logs",
        sample_size=args.sample_size,
        use_random=args.random,
        random_state=args.seed,
        sample_per_domain=args.sample_per_domain,
        scoring_mode=args.scoring,
        workers=args.workers,
        num_runs=args.repeat,
        use_validation=args.validation,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_api_key=llm_api_key,
        llm_temperature=args.llm_temperature,
        llm_top_p=args.llm_top_p,
        llm_max_tokens=args.llm_max_tokens,
    )


if __name__ == "__main__":
    main()
