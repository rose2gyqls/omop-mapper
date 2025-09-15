#!/usr/bin/env python3
"""
엔티티 매핑 API 단계별 테스트 스크립트
Adrenal Cushing's syndrome으로 각 단계별 로그 확인
"""

import sys
import os
import logging

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from omop_mapper.entity_mapping_api import EntityMappingAPI, EntityInput, DomainID

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def test_entity_mapping():
    """Adrenal Cushing's syndrome으로 엔티티 매핑 테스트"""
    
    print("=" * 80)
    print("🔬 엔티티 매핑 API 단계별 테스트")
    print("=" * 80)
    print(f"테스트 엔티티: Adrenal Cushing's syndrome")
    print("=" * 80)
    
    try:
        # EntityMappingAPI 인스턴스 생성
        api = EntityMappingAPI()
        
        # API 상태 확인
        health = api.health_check()
        print(f"📊 API 상태: {health}")
        print()
        
        # 테스트용 엔티티 입력 생성
        entity_input = EntityInput(
            entity_name="Adrenal Cushing's syndrome",
            domain_id=DomainID.CONDITION,  # Condition 도메인으로 설정
            vocabulary_id="SNOMED",
            confidence=1.0
        )
        
        print(f"📝 입력 엔티티 정보:")
        print(f"  - 엔티티 이름: {entity_input.entity_name}")
        print(f"  - 도메인 ID: {entity_input.domain_id.value}")
        print(f"  - 어휘체계 ID: {entity_input.vocabulary_id}")
        print(f"  - 신뢰도: {entity_input.confidence}")
        print()
        
        # 엔티티 매핑 실행
        print("🚀 엔티티 매핑 시작...")
        print()
        
        mapping_result = api.map_entity(entity_input)
        
        print()
        print("=" * 80)
        print("📋 최종 매핑 결과")
        print("=" * 80)
        
        if mapping_result:
            print(f"✅ 매핑 성공!")
            print(f"  - 매핑된 컨셉 ID: {mapping_result.mapped_concept_id}")
            print(f"  - 매핑된 컨셉 이름: {mapping_result.mapped_concept_name}")
            print(f"  - 도메인 ID: {mapping_result.domain_id}")
            print(f"  - 어휘체계 ID: {mapping_result.vocabulary_id}")
            print(f"  - 컨셉 클래스 ID: {mapping_result.concept_class_id}")
            print(f"  - 표준 컨셉: {mapping_result.standard_concept}")
            print(f"  - 컨셉 코드: {mapping_result.concept_code}")
            print(f"  - 매핑 점수: {mapping_result.mapping_score:.4f}")
            print(f"  - 매핑 신뢰도: {mapping_result.mapping_confidence}")
            print(f"  - 매핑 방법: {mapping_result.mapping_method}")
            
            if mapping_result.alternative_concepts:
                print(f"  - 대안 컨셉들 ({len(mapping_result.alternative_concepts)}개):")
                for i, alt in enumerate(mapping_result.alternative_concepts, 1):
                    print(f"    {i}. {alt['concept_name']} (ID: {alt['concept_id']}, 점수: {alt['score']:.4f})")
        else:
            print("❌ 매핑 실패")
        
        print()
        print("=" * 80)
        print("🏁 테스트 완료")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"테스트 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_entity_mapping()
