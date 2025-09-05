"""
Entity Mapping API - OMOP CDM 엔티티 매핑 라이브러리

이 라이브러리는 의료 엔티티를 OMOP CDM 표준 컨셉에 매핑하는 기능을 제공합니다.
Elasticsearch를 기반으로 한 고성능 유사도 검색과 Python 유사도 알고리즘을 통해
정확한 매핑을 수행합니다.

주요 기능:
- 엔티티 매핑: 의료 엔티티를 OMOP CDM 표준 컨셉에 매핑
- 다중 도메인 지원: Condition, Drug, Measurement, Procedure, Observation 등
- 유연한 검색: Elasticsearch 기반의 고성능 유사도 검색
- Standard/Non-standard 처리: Non-standard 컨셉을 Standard 컨셉으로 자동 변환
- 점수 기반 랭킹: Python 유사도 알고리즘을 통한 정확한 매핑

사용 예시:
    from entity_mapping_api import EntityMappingAPI, EntityInput, EntityTypeAPI
    
    # API 초기화
    api = EntityMappingAPI()
    
    # 엔티티 매핑
    entity_input = EntityInput(
        entity_name="diabetes mellitus",
        entity_type=EntityTypeAPI.DIAGNOSTIC
    )
    
    result = api.map_entity(entity_input)
    if result:
        print(f"매핑된 컨셉: {result.mapped_concept_name}")
        print(f"컨셉 ID: {result.mapped_concept_id}")
        print(f"매핑 점수: {result.mapping_score}")
"""

from .entity_mapping_api import (
    EntityMappingAPI,
    EntityInput,
    EntityTypeAPI,
    MappingResult,
    map_single_entity,
    map_entities_from_analysis,
    get_es_index
)

from .elasticsearch_client import ElasticsearchClient

__version__ = "1.0.0"
__author__ = "rose"
__email__ = "hyobinkim@gmail.com"

__all__ = [
    # Main API classes
    "EntityMappingAPI",
    "EntityInput", 
    "EntityTypeAPI",
    "MappingResult",
    
    # Utility functions
    "map_single_entity",
    "map_entities_from_analysis",
    "get_es_index",
    
    # Supporting classes
    "ElasticsearchClient",
]