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
from mapping_common import setup_logging, save_json, save_xlsx

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
    use_validation: bool = True,
):
    """엔티티 리스트로 매핑 실행.
    use_validation=False 이면 LLM validation 스킵, 출력: mapping_manual_noval_{timestamp}.*
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
    data_type_out = "manual_noval" if not use_validation else "manual"
    logger, _ = setup_logging(out_path, data_type_out, timestamp)

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

    save_json(results, out_path, data_type_out, timestamp)
    save_xlsx(results, out_path, data_type_out, timestamp)
    logger.info(f"결과: {out_path}/mapping_{data_type_out}_{timestamp}.(json|xlsx|log)")
    return results


if __name__ == "__main__":
    entity_list = [
        ("inflamed", "Condition", 4181063, "Inflammation of specific body organs")
    ]
    run_entity_list_test(entity_list, output_dir="test_logs", use_validation=True)
    # use_validation=False 로 비교 테스트: run_entity_list_test(entity_list, use_validation=False)
    print("완료.")
