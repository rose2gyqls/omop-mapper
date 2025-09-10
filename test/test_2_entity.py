"""
엔티티 매핑 API 테스트 코드
API 내부 함수들을 직접 사용하여 6단계별로 후보군과 점수를 확인
"""

import sys
import os
import logging
import numpy as np
from typing import Dict, Any, List

# 프로젝트 루트의 src 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from omop_mapper.entity_mapping_api import (
    EntityMappingAPI, 
    EntityInput, 
    DomainID
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_entity_mapping_6_steps(entity_name: str, domain_id: str, 
                               golden_concept_id: str, golden_concept_name: str) -> None:
    """
    엔티티 매핑을 6단계별로 상세 테스트 (API 함수 직접 사용)
    
    Args:
        entity_name: 테스트할 엔티티 이름
        domain_id: 도메인 ID
        golden_concept_id: 골든셋 컨셉 ID
        golden_concept_name: 골든셋 컨셉 이름
    """
    print(f"\n{'='*80}")
    print(f"🔍 엔티티: {entity_name}")
    print(f"🎯 골든셋: {golden_concept_id} - {golden_concept_name}")
    print(f"{'='*80}")
    
    # API 초기화
    api = EntityMappingAPI()
    
    # 엔티티 입력 생성
    entity_input = EntityInput(
        entity_name=entity_name,
        domain_id=DomainID(domain_id),
        confidence=1.0
    )
    
    # 사전 매핑 정보 준비
    entities_to_map = []
    entities_to_map.append({
        "entity_name": entity_input.entity_name,
        "domain_id": entity_input.domain_id or None,
        "vocabulary_id": entity_input.vocabulary_id or None
    })
    
    if not entities_to_map:
        print("❌ 엔티티 매핑 준비 실패")
        return
    
    entity_info = entities_to_map[0]
    
    # ===== 1단계: Elasticsearch 검색 =====
    print(f"\n🔍 1단계: Elasticsearch 검색")
    print(f"{'='*60}")
    
    candidates = api._search_similar_concepts(entity_input, entity_info, top_k=5)
    
    if not candidates:
        print("❌ 검색 결과 없음")
        return
    
    print(f"총 {len(candidates)}개 후보 발견:")
    for i, candidate in enumerate(candidates, 1):
        source = candidate['_source']
        score = candidate['_score']
        
        print(f"  {i}. Elasticsearch 점수: {score:.3f}")
        print(f"     컨셉 ID: {source.get('concept_id', 'N/A')}")
        print(f"     컨셉명: {source.get('concept_name', 'N/A')}")
        print(f"     도메인: {source.get('domain_id', 'N/A')}")
        print(f"     어휘체계: {source.get('vocabulary_id', 'N/A')}")
        print(f"     표준여부: {source.get('standard_concept', 'N/A')}")
        print()
    
    # ===== 2단계: Standard/Non-standard 분류 =====
    print(f"🔄 2단계: Standard/Non-standard 분류")
    print(f"{'='*60}")
    
    standard_candidates = []
    non_standard_candidates = []
    
    for i, candidate in enumerate(candidates, 1):
        source = candidate['_source']
        
        if source.get('standard_concept') == 'S':
            standard_candidates.append(candidate)
            print(f"  {i}. ✅ Standard: {source.get('concept_name', 'N/A')}")
        else:
            non_standard_candidates.append(candidate)
            print(f"  {i}. ⚠️ Non-standard: {source.get('concept_name', 'N/A')}")
    
    print(f"\n  📊 분류 결과: Standard {len(standard_candidates)}개, Non-standard {len(non_standard_candidates)}개")
    
    # ===== 3단계: Non-standard인 경우 → Maps to 관계로 Standard 후보 조회 =====
    print(f"\n🔗 3단계: Non-standard → Maps to 관계로 Standard 후보 조회")
    print(f"{'='*60}")
    
    # Non-standard 후보들은 Standard 후보 조회
    non_standard_to_standard_mappings = []
    
    for i, candidate in enumerate(non_standard_candidates, 1):
        source = candidate['_source']
        concept_id = str(source.get('concept_id', ''))
        print(f"  Non-standard {i}: {source.get('concept_name', 'N/A')} (ID: {concept_id})")
        
        standard_candidates_from_non = api._get_standard_candidates(concept_id, entity_info["domain_id"])
        
        if standard_candidates_from_non:
            print(f"    → Maps to 관계로 {len(standard_candidates_from_non)}개 Standard 후보 발견")
            for j, std_candidate in enumerate(standard_candidates_from_non[:3], 1):  # 상위 3개만 출력
                print(f"      {j}. {std_candidate.get('concept_name', 'N/A')} (ID: {std_candidate.get('concept_id', 'N/A')})")
            
            # 모든 Standard 후보들을 임시 저장 (나중에 유사도 재계산)
            non_standard_to_standard_mappings.append({
                'non_standard_source': source,
                'non_standard_candidate': candidate,
                'standard_candidates': standard_candidates_from_non
            })
        else:
            print(f"    → Maps to 관계 없음")
        
        print()
    
    # ===== 4단계: 모든 후보군에 대해 하이브리드 점수 기반 Re-ranking =====
    print(f"🎯 4단계: 모든 후보군에 대해 하이브리드 점수 기반 Re-ranking")
    print(f"{'='*60}")
    
    all_standard_candidates = []
    
    # 1. Standard 후보들에 대해 하이브리드 점수 재계산 (API와 동일한 방식)
    print("  📊 Standard 후보들 하이브리드 점수 재계산:")
    for i, candidate in enumerate(standard_candidates, 1):
        source = candidate['_source']
        original_score = candidate['_score']
        
        # API의 _calculate_hybrid_score 메서드 사용하여 일관된 점수 계산
        hybrid_score, text_sim, semantic_sim = api._calculate_hybrid_score(
            entity_name, 
            source.get('concept_name', ''),
            original_score,  # Elasticsearch 점수 전달 (하지만 API에서는 무시됨)
            source
        )
        
        print(f"    {i}. {source.get('concept_name', 'N/A')}")
        print(f"       하이브리드 점수: {hybrid_score:.3f} (텍스트: {text_sim:.3f}, 의미: {semantic_sim:.3f})")
        
        all_standard_candidates.append({
            'concept': source,
            'final_score': hybrid_score,  # 하이브리드 점수 사용
            'is_original_standard': True,
            'original_candidate': candidate,
            'elasticsearch_score': original_score,
            'hybrid_score': hybrid_score,
            'text_similarity': text_sim,
            'semantic_similarity': semantic_sim
        })
        print()
    
    # 2. Non-standard → Standard 후보들에 대해 하이브리드 점수 재계산
    print("  📊 Non-standard → Standard 후보들 하이브리드 점수 재계산:")
    for i, mapping in enumerate(non_standard_to_standard_mappings, 1):
        non_standard_source = mapping['non_standard_source']
        non_standard_candidate = mapping['non_standard_candidate']
        standard_candidates_list = mapping['standard_candidates']
        
        print(f"    Non-standard {i}: {non_standard_source.get('concept_name', 'N/A')}")
        
        for j, std_candidate in enumerate(standard_candidates_list, 1):
            # 하이브리드 점수 계산 (Non-standard → Standard의 경우 Elasticsearch 점수는 0으로 설정)
            hybrid_score, text_sim, semantic_sim = api._calculate_hybrid_score(
                entity_name, 
                std_candidate.get('concept_name', ''),
                0.0,  # Non-standard → Standard의 경우 Elasticsearch 점수 없음
                std_candidate
            )
            
            print(f"      Standard {j}: {std_candidate.get('concept_name', 'N/A')}")
            print(f"        하이브리드 점수: {hybrid_score:.3f} (텍스트: {text_sim:.3f}, 의미: {semantic_sim:.3f})")
            
            all_standard_candidates.append({
                'concept': std_candidate,
                'final_score': hybrid_score,  # 하이브리드 점수 사용
                'is_original_standard': False,
                'original_non_standard': non_standard_source,
                'original_candidate': non_standard_candidate,
                'hybrid_score': hybrid_score,
                'text_similarity': text_sim,
                'semantic_similarity': semantic_sim
            })
        
        print()
    
    # ===== 5단계: 점수 정규화 (0.0~1.0) → 최종 매핑 결과 =====
    print(f"📊 5단계: 점수 정규화 (0.0~1.0) → 최종 매핑 결과")
    print(f"{'='*60}")
    
    if not all_standard_candidates:
        print("❌ 처리된 후보 없음")
        return
    
    # 중복 제거 (동일한 concept_id와 concept_name인 경우 최고 점수만 유지)
    unique_candidates = {}
    for candidate in all_standard_candidates:
        concept = candidate['concept']
        concept_key = (concept.get('concept_id', ''), concept.get('concept_name', ''))
        
        # 동일한 컨셉이 이미 있는 경우 더 높은 점수만 유지
        if concept_key not in unique_candidates or candidate['final_score'] > unique_candidates[concept_key]['final_score']:
            unique_candidates[concept_key] = candidate
    
    # 중복 제거된 후보들을 리스트로 변환
    deduplicated_candidates = list(unique_candidates.values())
    
    print(f"중복 제거 전: {len(all_standard_candidates)}개 → 중복 제거 후: {len(deduplicated_candidates)}개")
    
    # 점수별 정렬 (하이브리드 점수 기준)
    sorted_candidates = sorted(deduplicated_candidates, key=lambda x: x['final_score'], reverse=True)
    
    print("최종 후보 순위 (하이브리드 점수 기준):")
    for i, candidate in enumerate(sorted_candidates, 1):
        concept = candidate['concept']
        final_score = candidate['final_score']  # 하이브리드 점수
        is_standard = candidate['is_original_standard']
        mapping_type = "직접 Standard" if is_standard else "Non-standard → Standard"
        
        # 점수 정규화 (하이브리드 점수는 이미 0~1 사이)
        normalized_score = api._normalize_score(final_score)
        confidence = api._determine_confidence(normalized_score)
        
        print(f"  {i}. 하이브리드 점수: {final_score:.3f} → 정규화: {normalized_score:.3f} ({confidence})")
        
        # 세부 점수 정보
        if 'hybrid_score' in candidate:
            text_sim = candidate.get('text_similarity', 0.0)
            semantic_sim = candidate.get('semantic_similarity', 0.0)
            print(f"     └ 텍스트: {text_sim:.3f} | 의미: {semantic_sim:.3f}")
        
        print(f"     컨셉 ID: {concept.get('concept_id', 'N/A')}")
        print(f"     컨셉명: {concept.get('concept_name', 'N/A')}")
        print(f"     도메인: {concept.get('domain_id', 'N/A')}")
        print(f"     어휘체계: {concept.get('vocabulary_id', 'N/A')}")
        print(f"     매핑 방법: {mapping_type}")
        
        print()
    
    # ===== 6단계: API의 map_entity 함수로 최종 매핑 결과 확인 =====
    print(f"🎯 6단계: API의 map_entity 함수로 최종 매핑 결과 확인")
    print(f"{'='*60}")
    
    # API의 map_entity 함수 호출
    mapping_result = api.map_entity(entity_input)
    
    if mapping_result:
        print(f"✅ API 매핑 성공!")
        print(f"   매핑된 컨셉 ID: {mapping_result.mapped_concept_id}")
        print(f"   매핑된 컨셉명: {mapping_result.mapped_concept_name}")
        print(f"   매핑 점수: {mapping_result.mapping_score:.3f}")
        print(f"   매핑 신뢰도: {mapping_result.mapping_confidence}")
        print(f"   매핑 방법: {mapping_result.mapping_method}")
        
        # 골든셋과 비교
        print(f"\n🎯 골든셋 비교:")
        if mapping_result.mapped_concept_id == golden_concept_id:
            print(f"  ✅ 성공! 골든셋과 정확히 일치")
        else:
            print(f"  ❌ 불일치")
        
        print(f"  예상: {golden_concept_id} - {golden_concept_name}")
        print(f"  실제: {mapping_result.mapped_concept_id} - {mapping_result.mapped_concept_name}")
    else:
        print("❌ API 매핑 실패")
    
    print(f"\n{'='*80}")


def main():
    """메인 테스트 함수"""
    print("🚀 엔티티 매핑 API 테스트 시작")
    
    # API 상태 확인
    api = EntityMappingAPI()
    health_check = api.health_check()
    print(f"📊 API 상태: {health_check}")
    
    # 테스트 케이스 1: Adrenal Cushing's syndrome
    print("\n" + "="*80)
    print("📋 테스트 케이스 1: Adrenal Cushing's syndrome")
    print("="*80)
    test_entity_mapping_6_steps(
        entity_name="Adrenal Cushing's syndrome",
        domain_id="condition",
        golden_concept_id="4030206",
        golden_concept_name="Adrenal Cushing's syndrome"
    )
    
    # 테스트 케이스 2: Acute Coronary Syndromes (ACS)
    print("\n" + "="*80)
    print("📋 테스트 케이스 2: Acute Coronary Syndromes (ACS)")
    print("="*80)
    test_entity_mapping_6_steps(
        entity_name="Acute Coronary Syndromes (ACS)",
        domain_id="condition",
        golden_concept_id="4215140",
        golden_concept_name="Acute coronary syndrome"
    )

    # 테스트 케이스 3: ST-segment elevation myocardial infarction (STEMI)
    print("\n" + "="*80)
    print("📋 테스트 케이스 3: ST-segment elevation myocardial infarction (STEMI)")
    print("="*80)
    test_entity_mapping_6_steps(
        entity_name="ST-segment elevation myocardial infarction (STEMI)",
        domain_id="condition",
        golden_concept_id="4296653",
        golden_concept_name="Acute ST segment elevation myocardial infarction"
    )
    
    print("\n🎉 모든 테스트 완료!")


if __name__ == "__main__":
    main()