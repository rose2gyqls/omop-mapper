"""
Stage 2: Non-standard to Standard 변환 및 중복 제거
- Standard 개념은 직접 추가
- Non-standard 개념은 "Maps to" 관계를 통해 Standard 개념으로 변환
- 동일한 concept_id와 concept_name인 경우 중복 제거
"""
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class Stage2StandardCollection:
    """Stage 2: Standard 후보 수집 및 중복 제거"""
    
    def __init__(self, es_client):
        """
        Args:
            es_client: Elasticsearch 클라이언트
        """
        self.es_client = es_client
    
    def collect_standard_candidates(
        self, 
        stage1_candidates: List[Dict[str, Any]], 
        domain_id: str
    ) -> List[Dict[str, Any]]:
        """
        Stage 1 후보들을 Standard 개념으로 변환하고 중복 제거
        
        **처리 로직**:
        1. Standard 개념 (S/C): 그대로 추가
        2. Non-standard 개념: concept-relationship 인덱스에서 "Maps to" 관계로 Standard 개념 조회
        3. 중복 제거: 동일한 concept_id와 concept_name인 경우 Elasticsearch 점수가 높은 것만 유지
        
        Args:
            stage1_candidates: Stage 1에서 검색된 후보들 (최대 9개)
            domain_id: 도메인 ID (디버깅용)
            
        Returns:
            List[Dict]: 중복 제거된 Standard 후보들
        """
        logger.info("=" * 80)
        logger.info("Stage 2: Non-standard → Standard 변환 및 중복 제거")
        logger.info("=" * 80)
        
        all_standard_candidates = []
        standard_count = 0
        non_standard_count = 0
        
        # 각 후보 처리
        for candidate in stage1_candidates:
            source = candidate['_source']
            search_type = candidate.get('_search_type', 'unknown')
            
            if source.get('standard_concept') in ['S', 'C']:
                # Standard 개념: 직접 추가
                standard_count += 1
                all_standard_candidates.append({
                    'concept': source,
                    'is_original_standard': True,
                    'original_candidate': candidate,
                    'elasticsearch_score': candidate['_score'],
                    'search_type': search_type
                })
                logger.info(f"  ✅ Standard: {source.get('concept_name', 'N/A')} "
                          f"(ID: {source.get('concept_id', 'N/A')}) "
                          f"[{search_type}]")
            else:
                # Non-standard 개념: Standard 개념으로 변환
                non_standard_count += 1
                concept_id = str(source.get('concept_id', ''))
                logger.info(f"  ⚠️ Non-standard: {source.get('concept_name', 'N/A')} "
                          f"(ID: {concept_id}) [{search_type}]")
                
                # "Maps to" 관계를 통해 Standard 개념 조회
                standard_candidates_from_non = self._get_standard_candidates(concept_id, domain_id)
                
                for std_candidate in standard_candidates_from_non:
                    all_standard_candidates.append({
                        'concept': std_candidate,
                        'is_original_standard': False,
                        'original_non_standard': source,
                        'original_candidate': candidate,
                        'elasticsearch_score': 0.0,  # Non-standard → Standard는 ES 점수 없음
                        'search_type': search_type
                    })
                    logger.info(f"     → Standard 매핑: {std_candidate.get('concept_name', 'N/A')} "
                              f"(ID: {std_candidate.get('concept_id', 'N/A')})")
        
        logger.info(f"\n📊 분류 결과:")
        logger.info(f"  - Standard: {standard_count}개")
        logger.info(f"  - Non-standard: {non_standard_count}개")
        logger.info(f"  - 총 수집된 Standard 후보: {len(all_standard_candidates)}개")
        
        # 중복 제거
        deduplicated_candidates = self._deduplicate_candidates(all_standard_candidates)
        
        logger.info(f"\n📊 중복 제거 완료: {len(all_standard_candidates)}개 → {len(deduplicated_candidates)}개")
        logger.info("=" * 80)
        
        return deduplicated_candidates
    
    def _get_standard_candidates(self, non_standard_concept_id: str, domain_id: str) -> List[Dict[str, Any]]:
        """
        Non-standard 개념의 Standard 후보들 조회
        
        Args:
            non_standard_concept_id: Non-standard 개념 ID
            domain_id: 도메인 ID
            
        Returns:
            List[Dict]: Standard 개념 후보들
        """
        try:
            # "Maps to" 관계 조회
            standard_concept_ids = self._get_maps_to_relationships(non_standard_concept_id)
            
            # Standard 개념 검색
            standard_candidates = self._search_concepts_in_all_indices(standard_concept_ids, domain_id)
            
            logger.debug(f"Non-standard {non_standard_concept_id}에 대한 "
                        f"{len(standard_candidates)}개 Standard 후보 조회 완료")
            return standard_candidates
            
        except Exception as e:
            logger.error(f"Standard 후보 조회 오류: {str(e)}")
            return []
    
    def _get_maps_to_relationships(self, concept_id_1: str) -> List[str]:
        """
        concept-relationship 인덱스에서 "Maps to" 관계 조회
        
        Args:
            concept_id_1: 소스 개념 ID
            
        Returns:
            List[str]: "Maps to"로 연결된 concept_id_2 리스트
        """
        try:
            relationship_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"concept_id_1": concept_id_1}},
                            {"match": {"relationship_id": "Maps to"}}
                        ]
                    }
                },
                "size": 10
            }
            
            relationship_response = self.es_client.es_client.search(
                index="concept-relationship",
                body=relationship_query
            )
            
            standard_concept_ids = []
            for hit in relationship_response['hits']['hits']:
                concept_id_2 = hit['_source'].get('concept_id_2')
                if concept_id_2:
                    standard_concept_ids.append(str(concept_id_2))
            
            if standard_concept_ids:
                logger.debug(f"concept-relationship 인덱스에서 {concept_id_1}에 대한 "
                           f"{len(standard_concept_ids)}개 Maps to 관계 발견: {standard_concept_ids}")
            
            return standard_concept_ids
            
        except Exception as e:
            logger.warning(f"concept-relationship 인덱스 Maps to 관계 조회 실패: {str(e)}")
            return []
    
    def _search_concepts_in_all_indices(self, concept_ids: List[str], domain_id: str) -> List[Dict[str, Any]]:
        """
        concept 인덱스에서 concept_id들 검색
        
        Args:
            concept_ids: 검색할 concept_id 리스트
            domain_id: 검색할 도메인 ID
            
        Returns:
            List[Dict]: 찾은 개념들
        """
        if not concept_ids:
            return []
        
        all_candidates = []
        
        try:
            concepts_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"terms": {"concept_id": concept_ids}},
                            {"terms": {"standard_concept": ["S", "C"]}}
                        ]
                    }
                },
                "size": len(concept_ids)
            }
            
            # Elasticsearch 클라이언트에 설정된 인덱스 사용
            concepts_response = self.es_client.es_client.search(
                index="concept-small",
                body=concepts_query
            )
            
            for hit in concepts_response['hits']['hits']:
                all_candidates.append(hit['_source'])
            
            if concepts_response['hits']['total']['value'] > 0:
                logger.debug(f"{concepts_response['hits']['total']['value']}개 Standard 개념 발견")
            
        except Exception as e:
            logger.warning(f"개념 검색 실패: {str(e)}")
        
        return all_candidates
    
    def _deduplicate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        중복 후보 제거 (동일한 concept_id와 concept_name인 경우)
        
        Args:
            candidates: 후보 리스트
            
        Returns:
            List[Dict]: 중복 제거된 후보 리스트
        """
        unique_candidates = {}
        
        for candidate in candidates:
            concept = candidate['concept']
            concept_key = (concept.get('concept_id', ''), concept.get('concept_name', ''))
            
            # 동일한 개념이 이미 있는 경우, 더 높은 Elasticsearch 점수만 유지
            if concept_key not in unique_candidates:
                unique_candidates[concept_key] = candidate
            else:
                existing_score = unique_candidates[concept_key]['elasticsearch_score']
                new_score = candidate['elasticsearch_score']
                
                # 더 높은 점수로 교체
                if new_score > existing_score:
                    unique_candidates[concept_key] = candidate
        
        return list(unique_candidates.values())

