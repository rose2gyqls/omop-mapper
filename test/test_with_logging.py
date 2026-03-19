"""
엔티티 리스트 기반 매핑 테스트 (run_mapping과 동일 출력 형식)

수동 엔티티 리스트로 테스트할 때 사용.
JSON, LOG, XLSX 동일 포맷으로 출력.
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

from datetime import datetime

from MapOMOP.entity_mapping_api import EntityMappingAPI, EntityInput, DomainID
from MapOMOP.elasticsearch_client import ElasticsearchClient
from mapping_common import setup_logging, save_json, save_xlsx, save_xlsx_repeat

DOMAIN_MAP = {
    "Condition": DomainID.CONDITION,
    "Procedure": DomainID.PROCEDURE,
    "Drug": DomainID.DRUG,
    "Observation": DomainID.OBSERVATION,
    "Measurement": DomainID.MEASUREMENT,
    "Device": DomainID.DEVICE,
}


def run_entity_list_test(
    entity_list: list,
    output_dir: str = "test_logs",
    scoring_mode: str = "llm",
    use_validation: bool = False,
    num_runs: int = 1,
):
    """엔티티 리스트로 매핑 실행.
    use_validation=False(기본): stage 1~3 점수 기반 최고 점수만 사용.
    use_validation=True: LLM validation 포함, 출력: mapping_manual_withval_{timestamp}.*
    num_runs > 1: 동일 엔티티로 N회 반복 (일관성 검증용). 현황+상세 시트 엑셀 저장.
    입력 형식:
      - (entity, domain)                          : domain만 지정
      - (entity, domain, gt_concept_id)            : ground truth ID 추가
      - (entity, domain, gt_concept_id, gt_name)   : ground truth ID, Name 추가
      - entity (str)                               : entity만, domain=None(전체 검색)
    """
    from tqdm import tqdm

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    data_type_out = "manual_withval" if use_validation else "manual"
    logger, _ = setup_logging(out_path, data_type_out, timestamp)

    num_runs = max(1, int(num_runs))
    logger.info(f"Scoring mode: {scoring_mode}, Runs: {num_runs}, Validation: {'on' if use_validation else 'off'}")

    all_results = []
    for run_idx in range(num_runs):
        if num_runs > 1:
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"매핑 Run {run_idx + 1}/{num_runs}")
            logger.info("=" * 80)

        es_client = ElasticsearchClient()
        es_client.concept_index = "concept-small"
        es_client.concept_synonym_index = "concept-synonym"
        api = EntityMappingAPI(
            es_client=es_client,
            scoring_mode=scoring_mode,
            use_validation=use_validation,
        )

        results = []
        for idx, item in enumerate(tqdm(entity_list, desc="매핑")):
            ground_truth_concept_id = None
            ground_truth_concept_name = None
            if isinstance(item, tuple) and len(item) >= 2:
                entity_name = str(item[0])
                domain_str = item[1]
                domain_id = DOMAIN_MAP.get(domain_str) if domain_str else None
                if len(item) >= 3:
                    ground_truth_concept_id = item[2]  # int or None
                if len(item) >= 4:
                    ground_truth_concept_name = item[3]
            else:
                entity_name = str(item)
                domain_id = None

            entity_input = EntityInput(entity_name=entity_name, domain_id=domain_id, vocabulary_id=None)
            mapping_results = api.map_entity(entity_input)

            stage1 = getattr(api, "_last_stage1_candidates", []) or []
            stage2 = getattr(api, "_last_stage2_candidates", []) or []
            stage3 = getattr(api, "_last_rerank_candidates", []) or []

            best = max(mapping_results, key=lambda x: x.mapping_score) if mapping_results else None
            mapping_correct = False
            if best and ground_truth_concept_id is not None:
                try:
                    mapping_correct = int(best.mapped_concept_id) == int(ground_truth_concept_id)
                except (ValueError, TypeError):
                    pass

            r = {
                "test_index": idx + 1,
                "id": f"manual_{idx + 1}",
                "entity_name": entity_name,
                "input_domain": domain_id.value if domain_id else "All",
                "ground_truth_concept_id": ground_truth_concept_id,
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

        all_results.append(results)

        # num_runs > 1: 각 run 완료 시마다 JSON/XLSX 즉시 저장 (전체 완료 대기 불필요)
        if num_runs > 1:
            completed_runs = len(all_results)
            save_json({"num_runs": completed_runs, "runs": all_results}, out_path, data_type_out, timestamp)
            save_xlsx_repeat(all_results, out_path, data_type_out, timestamp)
            logger.info(f"Run {completed_runs}/{num_runs} 완료 → JSON/XLSX 저장됨")

    total = len(all_results[-1])
    if num_runs > 1 and total > 0:
        all_same_count = sum(
            1 for row_idx in range(total)
            if len(set(str(all_results[run][row_idx].get("best_concept_id") or "") for run in range(num_runs))) == 1
        )
        logger.info(f"{num_runs}회 동일 결과: {all_same_count}/{total}개 ({100 * all_same_count / total:.2f}%)")

    if num_runs > 1:
        # 이미 각 run 완료 시 저장됨
        pass
    else:
        save_json(all_results[-1], out_path, data_type_out, timestamp)
        save_xlsx(all_results[-1], out_path, data_type_out, timestamp)
    logger.info(f"결과: {out_path}/mapping_{data_type_out}_{timestamp}.(json|xlsx|log)")
    return all_results if num_runs > 1 else all_results[-1]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="엔티티 리스트 기반 매핑 테스트")
    parser.add_argument("--repeat", "-r", type=int, default=1, help="동일 엔티티로 N회 반복 (기본: 1). 2+ 입력 시 현황+상세 시트 엑셀 저장")
    parser.add_argument("--output-dir", "-o", default="test_logs", help="출력 디렉터리")
    parser.add_argument("--validation", action="store_true", help="LLM validation 포함")
    args = parser.parse_args()

    entity_list = [
        ("atrial arrhythmia", "Condition"),
        # ("Prednicarbate 0.25% 60g lotion", "Drug", 35605482, "2 ml ondansetron 2 mg/ml injection"),
        # ("inflamed", "Condition", 4181063, "Inflammation of specific body organs"),
        # ("Pseudomyxoma peritonei", "Condition", 4146018, "pseudomyxoma peritonei"),
        # ("Platelet concentrate (400ml)", "Procedure", 4035234, "transfusion of platelet concentrate"),
        # ("Osteochondroma", "Condition", 40480080, "osteochondroma"),
        # ("Esomeprazole 40mg tab", "Drug", 19101745, "esomeprazole 40 mg oral tablet"),
        # ("Chlorpheniramine 4mg/2mL inj", "Drug", 42922061, "2 ml chlorpheniramine 2 mg/ml injectable solution"),
        # ("Pituitary adenoma, nonfunctioning", "Condition", 4112967, "functionless pituitary adenoma"),
        # ("Diclofenac 0.1% 5ml oph", "Drug", 21068327, "5 ml diclofenac 1 mg/ml ophthalmic solution"),
        # ("Chest PA", "Procedure", 36713260, "plain x-ray of chest, posteroanterior"),
        # ("advanced gastric cancer, adenocarcinoma", "Condition", 4248802, "adenocarcinoma of stomach")
    ]
    run_entity_list_test(
        entity_list,
        output_dir=args.output_dir,
        num_runs=args.repeat,
        use_validation=args.validation,
    )
    print("완료.")
