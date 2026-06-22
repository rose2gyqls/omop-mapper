#!/usr/bin/env python3
"""
Unified mapping execution script

Selecting a data source (snuh, snomed, etc.) applies the default CSV path and preprocessing.
Generates three files (JSON (raw), LOG, XLSX) in test_logs/ with the same timestamp.

Usage:
    python scripts/run_mapping.py snuh
    python scripts/run_mapping.py snomed
    python scripts/run_mapping.py snuh --sample-per-domain 5 --random
    python scripts/run_mapping.py snuh --workers 4   # parallel processing
"""

import argparse
import functools
import os
import sys
import time

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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

# Global for ProcessPoolExecutor workers (initialized in each process)
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
    """Worker process initialization: create the API instance once + set up logging.
    capture_only=True: put logs in capture mode (grouped per entity for output when multiple workers).
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
    """Map a single entity (runs in a worker, uses only picklable arguments).
    capture_logs=True: capture API logs and return (test_index, result, log_lines).
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
    """Run mapping: load data (default path + preprocessing) -> map -> output JSON/LOG/XLSX.
    If workers > 1, process in parallel with ProcessPoolExecutor.
    If num_runs > 1, repeat N times on the same data (for consistency verification).
    use_validation=False (default): use only the highest-scoring mapping based on stage 1-3 scores.
    use_validation=True: include the LLM validation module. The output file name gets _withval appended.
    """
    from datetime import datetime
    from tqdm import tqdm

    if data_type not in DATA_SOURCES:
        raise ValueError(f"Unknown data_type: {data_type}. Available: {list(DATA_SOURCES.keys())}")

    config = DATA_SOURCES[data_type]
    csv_path = config["csv_path"]
    id_col = config["id_col"]

    # sample_per_domain: integer N -> N per domain
    sample_per_domain_dict = None
    if sample_per_domain is not None:
        sample_per_domain_dict = {d: sample_per_domain for d in config["domains"]}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data_type_out = f"{data_type}_withval" if use_validation else data_type
    # workers > 1: terminal shows only tqdm progress, detailed logs go to a file
    logger, log_file = setup_logging(out_path, data_type_out, timestamp, console=(workers == 1))
    logger.info("=" * 80)
    logger.info(f"Mapping started: data={data_type}, csv={csv_path}")
    logger.info(f"Preprocessing: {config.get('vocabulary_filter', config.get('filter_domains', 'none'))}")
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

    # Load data (with preprocessing applied)
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
    logger.info(f"Loaded data: {len(df)} rows")
    logger.info(f"Scoring mode: {scoring_mode}, Workers: {workers}, Runs: {num_runs}, Validation: {'on' if use_validation else 'off'}")

    all_results = []  # when num_runs > 1: [run1_results, run2_results, ...]
    start_time = time.time()

    for run_idx in range(num_runs):
        if num_runs > 1:
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"Mapping Run {run_idx + 1}/{num_runs}")
            logger.info("=" * 80)

        results = []

        # Parallel: ProcessPoolExecutor (workers > 1)
        if workers > 1:
            tasks = []
            for idx, row in df.iterrows():
                entity_name, domain_id, record_id, ground_truth, ground_truth_concept_name = row_to_input(row, DOMAIN_MAP)
                domain_str = domain_id.value if domain_id else None
                # SNOMED: use test_index from CSV; otherwise: row index + 1
                csv_test_index = int(row["test_index"]) if "test_index" in df.columns and pd.notna(row.get("test_index")) else (idx + 1)
                tasks.append((idx, csv_test_index, entity_name, domain_str, record_id, ground_truth, ground_truth_concept_name, id_col))

            completed = 0
            log_buffer = {}  # row_idx -> log_lines (stored in entity order, not completion order)
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
                with tqdm(total=len(tasks), desc="Mapping", unit="item") as pbar:
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
                                status = "Correct" if r.get("mapping_correct") else ("Incorrect" if r.get("success") else "Failed")
                                logger.info(f"#{r.get('test_index')} {r['entity_name']}: {status}")
                            pbar.update(1)
                        except Exception as e:
                            task = tasks[row_idx]
                            test_index = task[1]
                            logger.error(f"#{test_index} Worker exception: {e}")
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
                # Logs: write to the file in entity order (row_idx)
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
            # Sequential processing (workers == 1)
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
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Mapping"):
                try:
                    entity_name, domain_id, record_id, ground_truth, ground_truth_concept_name = row_to_input(row, DOMAIN_MAP)
                    # SNOMED: use test_index from CSV; otherwise: row index + 1
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
                        status = "Correct" if mapping_correct else ("Incorrect" if r["success"] else "Failed")
                        logger.info(f"#{csv_test_index} {entity_name}: {status}")

                except Exception as e:
                    csv_test_index = int(row["test_index"]) if "test_index" in df.columns and pd.notna(row.get("test_index")) else (idx + 1)
                    logger.error(f"#{csv_test_index} Error: {e}")
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

        # num_runs > 1: save JSON/XLSX immediately after each run completes (no need to wait for all to finish)
        if num_runs > 1:
            completed_runs = len(all_results)
            save_json({"num_runs": completed_runs, "runs": all_results}, out_path, data_type_out, timestamp)
            save_xlsx_repeat(all_results, out_path, data_type_out, timestamp)
            logger.info(f"Run {completed_runs}/{num_runs} complete -> JSON/XLSX saved")

    elapsed = time.time() - start_time

    # Summary: based on the last run (same format as a single run)
    results = all_results[-1]
    total = len(results)
    success_count = sum(1 for r in results if r["success"])
    correct_count = sum(1 for r in results if r["mapping_correct"])

    logger.info("")
    logger.info("=" * 80)
    logger.info("Results summary")
    logger.info("=" * 80)
    logger.info(f"Total: {total}, Success: {success_count} ({100 * success_count / total:.2f}%), Correct: {correct_count} ({100 * correct_count / total:.2f}%)")
    logger.info(f"Elapsed: {elapsed:.2f}s ({elapsed / 60:.2f}min)")

    if workers > 1:
        print(f"\nTotal {total}, success {success_count} ({100 * success_count / total:.1f}%), correct {correct_count} | elapsed {elapsed / 60:.1f}min")

    if num_runs > 1:
        all_same_count = 0
        for row_idx in range(total):
            concept_ids = [
                str(all_results[run][row_idx].get("best_concept_id") or "")
                for run in range(num_runs)
            ]
            if len(set(concept_ids)) == 1:
                all_same_count += 1
        logger.info(f"Identical results across {num_runs} runs: {all_same_count}/{total} ({100 * all_same_count / total:.2f}%)")
        # Already saved after each run completes. Only log the final paths
        json_path = out_path / f"mapping_{data_type_out}_{timestamp}.json"
        xlsx_path = out_path / f"mapping_{data_type_out}_{timestamp}.xlsx"
    else:
        json_path = save_json(results, out_path, data_type_out, timestamp)
        xlsx_path = save_xlsx(results, out_path, data_type_out, timestamp)
    logger.info(f"JSON: {json_path}")
    logger.info(f"XLSX: {xlsx_path}")

    return all_results if num_runs > 1 else results


def main():
    parser = argparse.ArgumentParser(description="OMOP mapping unified runner")
    parser.add_argument(
        "data",
        choices=list(DATA_SOURCES.keys()),
        help=f"Data source (default CSV and preprocessing applied automatically). Available: {list(DATA_SOURCES.keys())}",
    )
    parser.add_argument("--sample-size", "-n", type=int, default=None, help="Sample size (uses all data if -n not given; applied when sample-per-domain is not used)")
    parser.add_argument("--random", action="store_true", help="Random sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--sample-per-domain",
        type=int,
        default=None,
        metavar="N",
        help="Sample N per domain. Example: --sample-per-domain 5",
    )
    parser.add_argument(
        "--scoring",
        default="llm",
        choices=["llm", "llm_with_score", "semantic"],
        help="Scoring mode (for ablation study)",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "together"],
        default=None,
        help="LLM route selection. Uses LLM_PROVIDER/env default if not specified.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Model name override. Together also supports gpt_oss_20b, mistral_small_24b, llama4_maverick aliases.",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="OpenAI-compatible base URL override.",
    )
    parser.add_argument(
        "--llm-api-key-env",
        default=None,
        help="Name of the environment variable to read the API key from. Example: OPENAI_API_KEY",
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
        help="LLM max output tokens override. Uses env/default if not specified.",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel workers (default: 1, sequential processing). 4-8 recommended.",
    )
    parser.add_argument(
        "--repeat",
        "-r",
        type=int,
        default=1,
        metavar="N",
        help="Repeat mapping N times on the same data (for consistency verification). Entering 5 generates a summary + 5 detail sheets.",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Include the LLM validation module (default: use only the highest score based on stage 1-3 scores). Output: mapping_{snuh|snomed}_withval_{timestamp}.*",
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
