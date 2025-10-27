from pickle import NONE
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
import logging
from enum import Enum

from .elasticsearch_client import ElasticsearchClient

try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SAPBERT = True
except ImportError:
    HAS_SAPBERT = False

logger = logging.getLogger(__name__)


class DomainID(Enum):
    """도메인 ID"""
    PROCEDURE = "procedure"
    CONDITION = "condition"
    DRUG = "drug"
    OBSERVATION = "observation"
    MEASUREMENT = "measurement"
    THRESHOLD = "threshold"
    DEMOGRAPHICS = "demographics"
    PERIOD = "period"
    PROVIDER = "provider"


@dataclass
class EntityInput:
    """입력용 엔티티 데이터"""
    entity_name: str
    domain_id: DomainID
    vocabulary_id: Optional[str] = None


@dataclass
class MappingResult:
    """매핑 결과 데이터"""
    source_entity: EntityInput
    mapped_concept_id: str
    mapped_concept_name: str
    domain_id: str
    vocabulary_id: str
    concept_class_id: str
    standard_concept: str
    concept_code: str
    concept_embedding: List[float]
    valid_start_date: Optional[str] = None
    valid_end_date: Optional[str] = None
    invalid_reason: Optional[str] = None
    mapping_score: float = 0.0
    mapping_confidence: str = "low"
    mapping_method: str = "unknown"
    alternative_concepts: List[Dict[str, Any]] = None
    

class EntityMappingAPI:
    """엔티티 매핑 API 클래스"""

    def __init__(
        self,
        es_client: Optional[ElasticsearchClient] = None,
        confidence_threshold: float = 0.5
    ):
        """
        엔티티 매핑 API 초기화
        
        Args:
            es_client: Elasticsearch 클라이언트
            confidence_threshold: 매핑 신뢰도 임계치
        """
        self.es_client = es_client or ElasticsearchClient.create_default()
        self.confidence_threshold = confidence_threshold
    
    def map_entity(self, entity_input: EntityInput) -> Optional[MappingResult]:
        """
        단일 엔티티를 OMOP CDM에 3단계 매핑
        1단계: Elasticsearch 쿼리로 top 5개 후보군 추출
        2단계: Standard/Non-standard 분류 및 모든 Standard 후보군 수집 후 중복 제거
        3단계: 수집된 후보군들에 대해 모두 hybrid 점수(concept_embedding 필드 사용)로 계산
        
        Args:
            entity_input: 매핑할 엔티티 정보
            
        Returns:
            MappingResult: 매핑 결과 또는 None (매핑 실패시)
        """
        try:
            entity_name = entity_input.entity_name
            domain_id = entity_input.domain_id
            
            logger.info(f"🚀 3단계 엔티티 매핑 시작: {entity_name} (도메인: {domain_id})")
            
            # ===== 1단계: Elasticsearch 쿼리로 top 5개 후보군 추출 =====
            stage1_candidates = self._stage1_elasticsearch_search(entity_input)
            if not stage1_candidates:
                logger.warning(f"⚠️ 1단계 실패 - 검색 결과 없음: {entity_name}")
                return None
            
            # ===== 2단계: Standard/Non-standard 분류 및 모든 Standard 후보군 수집 후 중복 제거 =====
            stage2_candidates = self._stage2_collect_standard_candidates(stage1_candidates, domain_id)
            if not stage2_candidates:
                logger.warning(f"⚠️ 2단계 실패 - Standard 후보 없음: {entity_name}")
                return None
            
            # ===== 3단계: 수집된 후보군들에 대해 모두 hybrid 점수 계산 =====
            stage3_candidates = self._stage3_calculate_hybrid_scores(entity_input, stage2_candidates)
            if not stage3_candidates:
                logger.warning(f"⚠️ 3단계 실패 - 점수 계산 실패: {entity_name}")
                return None
            
            # 최종 매핑 결과 생성
            mapping_result = self._create_final_mapping_result(entity_input, stage3_candidates)
            
            logger.info(f"✅ 3단계 매핑 성공: {entity_name} -> {mapping_result.mapped_concept_name}")
            logger.info(f"📊 최종 매핑 점수: {mapping_result.mapping_score:.4f} (신뢰도: {mapping_result.mapping_confidence})")
            return mapping_result
                
        except Exception as e:
            logger.error(f"⚠️ 엔티티 매핑 오류: {str(e)}")
            return None
    
    def _stage1_elasticsearch_search(self, entity_input: EntityInput) -> List[Dict[str, Any]]:
        """
        1단계: Elasticsearch 쿼리로 top 5개 후보군 추출
        디버깅용으로 벡터 검색, 텍스트 검색, 하이브리드 검색을 각각 수행하여 결과 비교
        
        Args:
            entity_input: 엔티티 입력 정보
            
        Returns:
            List[매칭된 컨셉 후보들] - 하이브리드 검색 결과
        """
        logger.info("=" * 60)
        logger.info("1단계: Elasticsearch 쿼리로 top 5개 후보군 추출 (디버깅 모드)")
        logger.info("=" * 60)
        
        entity_name = entity_input.entity_name
        domain_id = entity_input.domain_id
        es_index = "concept"
        top_k = 5
        
        logger.info(f"🔍 엔티티: {entity_name}, 도메인: {domain_id}")
        
        # 엔티티 임베딩 생성
        entity_embedding = None
        if HAS_SAPBERT:
            entity_embedding = self._get_simple_embedding(entity_name)
            if entity_embedding is not None:
                logger.info("✅ 엔티티 임베딩 생성 성공")
            else:
                logger.warning("⚠️ 엔티티 임베딩 생성 실패")
        else:
            logger.warning("⚠️ SapBERT 미설치")
        
        # 1. 벡터 검색만 수행 (디버깅용)
        logger.info("\n" + "=" * 40)
        logger.info("🧠 1-1. 벡터 검색 결과")
        logger.info("=" * 40)
        vector_results = []
        if entity_embedding is not None:
            vector_results = self._perform_vector_search_silent(entity_embedding, es_index, top_k)
            logger.info(f"벡터 검색 결과: {len(vector_results)}개")
            for i, hit in enumerate(vector_results, 1):
                source = hit['_source']
                logger.info(f"  {i}. {source.get('concept_name', 'N/A')} "
                          f"(ID: {source.get('concept_id', 'N/A')}) "
                          f"- 벡터 점수: {hit['_score']:.4f}")
        else:
            logger.info("벡터 검색 건너뜀 (임베딩 없음)")
        
        # 2. 텍스트 검색만 수행 (디버깅용)
        logger.info("\n" + "=" * 40)
        logger.info("📝 1-2. 텍스트 검색 결과")
        logger.info("=" * 40)
        text_results = self._perform_text_only_search_silent(entity_name, es_index, top_k)
        logger.info(f"텍스트 검색 결과: {len(text_results)}개")
        for i, hit in enumerate(text_results, 1):
            source = hit['_source']
            logger.info(f"  {i}. {source.get('concept_name', 'N/A')} "
                      f"(ID: {source.get('concept_id', 'N/A')}) "
                      f"- 텍스트 점수: {hit['_score']:.4f}")
        
        # 3. 하이브리드 검색 수행 (최종 결과용)
        logger.info("\n" + "=" * 40)
        logger.info("🔄 1-3. 하이브리드 검색 결과 (최종)")
        logger.info("=" * 40)
        
        if entity_embedding is not None:
            # 벡터+텍스트 하이브리드 쿼리 수행
            hybrid_results = self._perform_native_hybrid_search(entity_name, entity_embedding, es_index, top_k)
        else:
            # 텍스트만 사용
            hybrid_results = text_results
        
        logger.info(f"하이브리드 검색 결과: {len(hybrid_results)}개")
        for i, hit in enumerate(hybrid_results, 1):
            source = hit['_source']
            standard_status = "Standard" if source.get('standard_concept') in ['S', 'C'] else "Non-standard"
            concept_name = source.get('concept_name', 'N/A')
            concept_length = len(concept_name) if concept_name != 'N/A' else 0
            length_diff = abs(len(entity_name.strip()) - concept_length)
            logger.info(f"  {i}. {concept_name} "
                      f"(ID: {source.get('concept_id', 'N/A')}) "
                      f"- {standard_status}, 하이브리드 점수: {hit['_score']:.4f}")
        
        logger.info(f"\n📊 1단계 최종 결과: {len(hybrid_results)}개 후보 (하이브리드 검색)")
        
        # 디버깅용: stage1 후보군 저장
        self._last_stage1_candidates = [
            {
                'concept_id': str(hit['_source'].get('concept_id', '')),
                'concept_name': hit['_source'].get('concept_name', ''),
                'vocabulary_id': hit['_source'].get('vocabulary_id', ''),
                'standard_concept': hit['_source'].get('standard_concept', ''),
                'elasticsearch_score': hit['_score']
            }
            for hit in hybrid_results
        ]
        
        return hybrid_results
    
    def _stage2_collect_standard_candidates(self, stage1_candidates: List[Dict[str, Any]], domain_id: str) -> List[Dict[str, Any]]:
        """
        2단계: Standard/Non-standard 분류 및 모든 Standard 후보군 수집 후 중복 제거
        
        Args:
            stage1_candidates: 1단계에서 검색된 후보들
            domain_id: 도메인 ID
            
        Returns:
            List[중복 제거된 Standard 후보들]
        """
        logger.info("=" * 60)
        logger.info("2단계: Standard/Non-standard 분류 및 모든 Standard 후보군 수집")
        logger.info("=" * 60)
        
        all_standard_candidates = []
        standard_count = 0
        non_standard_count = 0
        
        for candidate in stage1_candidates:
            source = candidate['_source']
            
            if source.get('standard_concept') == 'S' or source.get('standard_concept') == 'C':
                # Standard 엔티티: 직접 추가
                standard_count += 1
                all_standard_candidates.append({
                    'concept': source,
                    'is_original_standard': True,
                    'original_candidate': candidate,
                    'elasticsearch_score': candidate['_score']
                })
                logger.info(f"  Standard 추가: {source.get('concept_name', 'N/A')} (concept_id: {source.get('concept_id', 'N/A')})")
            else:
                # Non-standard 엔티티: Standard 후보들 조회 후 추가
                non_standard_count += 1
                concept_id = str(source.get('concept_id', ''))
                logger.info(f"  Non-standard 처리: {source.get('concept_name', 'N/A')} (concept_id: {concept_id})")
                
                standard_candidates_from_non = self._get_standard_candidates(concept_id, domain_id)
                
                for std_candidate in standard_candidates_from_non:
                    all_standard_candidates.append({
                        'concept': std_candidate,
                        'is_original_standard': False,
                        'original_non_standard': source,
                        'original_candidate': candidate,
                        'elasticsearch_score': 0.0  # Non-standard → Standard의 경우 Elasticsearch 점수 없음
                    })
                    logger.info(f"    -> Standard 매핑: {std_candidate.get('concept_name', 'N/A')} (concept_id: {std_candidate.get('concept_id', 'N/A')})")
        
        logger.info(f"📊 2단계 분류 결과: Standard {standard_count}개, Non-standard {non_standard_count}개")
        logger.info(f"📊 수집된 총 Standard 후보: {len(all_standard_candidates)}개")
        
        # 중복 제거 (동일한 concept_id와 concept_name인 경우 최고 Elasticsearch 점수만 유지)
        unique_candidates = {}
        for candidate in all_standard_candidates:
            concept = candidate['concept']
            concept_key = (concept.get('concept_id', ''), concept.get('concept_name', ''))
            
            # 동일한 컨셉이 이미 있는 경우 더 높은 Elasticsearch 점수만 유지
            if concept_key not in unique_candidates or candidate['elasticsearch_score'] > unique_candidates[concept_key]['elasticsearch_score']:
                unique_candidates[concept_key] = candidate
        
        # 중복 제거된 후보들을 리스트로 변환
        deduplicated_candidates = list(unique_candidates.values())
        
        logger.info(f"📊 중복 제거 완료: {len(all_standard_candidates)}개 → {len(deduplicated_candidates)}개 후보")
        
        return deduplicated_candidates
    
    def _stage3_calculate_hybrid_scores(self, entity_input: EntityInput, stage2_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        3단계: 수집된 후보군들에 대해 모두 hybrid 점수(concept_embedding 필드 사용)로 계산
        
        Args:
            entity_input: 엔티티 입력 정보
            stage2_candidates: 2단계에서 수집된 Standard 후보들
            
        Returns:
            List[hybrid 점수가 계산된 후보들 (점수 순으로 정렬)]
        """
        logger.info("=" * 60)
        logger.info("3단계: 수집된 후보군들에 대해 모두 hybrid 점수 계산")
        logger.info("=" * 60)
        
        final_candidates = []
        
        for i, candidate in enumerate(stage2_candidates, 1):
            concept = candidate['concept']
            elasticsearch_score = candidate['elasticsearch_score']
            
            logger.info(f"  {i}. {concept.get('concept_name', 'N/A')} (concept_id: {concept.get('concept_id', 'N/A')})")
            logger.info(f"     Elasticsearch 점수: {elasticsearch_score:.4f}")
            
            # 하이브리드 점수 계산 (텍스트 + 의미적 유사도, concept_embedding 필드 사용)
            hybrid_score, text_sim, semantic_sim = self._calculate_hybrid_score(
                entity_input.entity_name, 
                concept.get('concept_name', ''),
                elasticsearch_score,
                concept
            )
            
            logger.info(f"     텍스트 유사도: {text_sim:.4f}")
            logger.info(f"     의미적 유사도: {semantic_sim:.4f}")
            logger.info(f"     하이브리드 점수: {hybrid_score:.4f}")
            
            final_candidates.append({
                'concept': concept,
                'final_score': hybrid_score,
                'is_original_standard': candidate['is_original_standard'],
                'original_candidate': candidate['original_candidate'],
                'elasticsearch_score': elasticsearch_score,
                'text_similarity': text_sim,
                'semantic_similarity': semantic_sim
            })
        
        # 하이브리드 점수 기준으로 정렬
        sorted_candidates = sorted(final_candidates, key=lambda x: x['final_score'], reverse=True)
        
        logger.info("📊 3단계 결과 - 하이브리드 점수 순위:")
        for i, candidate in enumerate(sorted_candidates, 1):
            concept = candidate['concept']
            logger.info(f"  {i}. {concept.get('concept_name', 'N/A')} "
                      f"(concept_id: {concept.get('concept_id', 'N/A')}) "
                      f"- 점수: {candidate['final_score']:.4f} "
                      f"(텍스트: {candidate['text_similarity']:.4f}, "
                      f"의미적: {candidate['semantic_similarity']:.4f})")
        
        # 디버깅용: 마지막 리랭킹 후보 저장 (stage3)
        self._last_rerank_candidates = [
            {
                'concept_id': str(c['concept'].get('concept_id', '')),
                'concept_name': c['concept'].get('concept_name', ''),
                'vocabulary_id': c['concept'].get('vocabulary_id', ''),
                'standard_concept': c['concept'].get('standard_concept', ''),
                'elasticsearch_score': c.get('elasticsearch_score', 0.0),
                'text_similarity': c.get('text_similarity', 0.0),
                'semantic_similarity': c.get('semantic_similarity', 0.0),
                'final_score': c.get('final_score', 0.0)
            }
            for c in sorted_candidates
        ]
        
        return sorted_candidates
    
    def _create_final_mapping_result(self, entity_input: EntityInput, sorted_candidates: List[Dict[str, Any]]) -> MappingResult:
        """
        최종 매핑 결과 생성
        
        Args:
            entity_input: 원본 엔티티 입력
            sorted_candidates: 점수 순으로 정렬된 후보들
            
        Returns:
            MappingResult: 매핑 결과
        """
        best_candidate = sorted_candidates[0]
        alternative_candidates = sorted_candidates[1:4]  # 상위 3개 대안
        
        mapping_result = self._create_mapping_result(entity_input, best_candidate, alternative_candidates)
        
        mapping_type = "direct_standard" if best_candidate['is_original_standard'] else "non_standard_to_standard"
        logger.info(f"📊 매핑 유형: {mapping_type}")
        
        return mapping_result
    
    
    
    def _perform_vector_search(self, entity_embedding: np.ndarray, es_index: str, top_k: int) -> List[Dict[str, Any]]:
        """
        벡터 검색 수행 (knn 쿼리만 사용)
        
        Args:
            entity_embedding: 엔티티 임베딩 벡터
            es_index: Elasticsearch 인덱스
            top_k: 반환할 결과 수
            
        Returns:
            List[벡터 검색 결과들]
        """
        logger.info(f"🧠 벡터 검색 수행")
        
        # 임베딩을 리스트로 변환
        embedding_list = entity_embedding.tolist()
        
        # knn 쿼리만 사용
        vector_query = {
            "knn": {
                "field": "concept_embedding",
                "query_vector": embedding_list,
                "k": top_k,
                "num_candidates": top_k * 3
            },
            "size": top_k,
            "_source": True
        }
        
        try:
            response = self.es_client.es_client.search(
                index=es_index,
                body=vector_query
            )
            
            hits = response['hits']['hits'] if response['hits']['total']['value'] > 0 else []
            logger.info(f"✅ 벡터 검색 완료: {len(hits)}개 결과")
            
            # 모든 결과 로깅
            for i, hit in enumerate(hits, 1):
                source = hit['_source']
                logger.info(f"  {i}. {source.get('concept_name', 'N/A')} "
                          f"(ID: {source.get('concept_id', 'N/A')}) "
                          f"- 벡터 점수: {hit['_score']:.4f}")
            
            return hits
            
        except Exception as e:
            logger.error(f"벡터 검색 실패: {e}")
            return []
    
    def _perform_vector_search_silent(self, entity_embedding: np.ndarray, es_index: str, top_k: int) -> List[Dict[str, Any]]:
        """
        벡터 검색 수행 (로깅 없는 버전)
        
        Args:
            entity_embedding: 엔티티 임베딩 벡터
            es_index: Elasticsearch 인덱스
            top_k: 반환할 결과 수
            
        Returns:
            List[벡터 검색 결과들]
        """
        # 임베딩을 리스트로 변환
        embedding_list = entity_embedding.tolist()
        
        # knn 쿼리만 사용
        vector_query = {
            "knn": {
                "field": "concept_embedding",
                "query_vector": embedding_list,
                "k": top_k,
                "num_candidates": top_k * 3
            },
            "size": top_k,
            "_source": True
        }
        
        try:
            response = self.es_client.es_client.search(
                index=es_index,
                body=vector_query
            )
            
            hits = response['hits']['hits'] if response['hits']['total']['value'] > 0 else []
            return hits
            
        except Exception as e:
            logger.error(f"벡터 검색 실패: {e}")
            return []
    
    
    def _perform_text_only_search(self, entity_name: str, es_index: str, top_k: int) -> List[Dict[str, Any]]:
        """
        간단한 텍스트 검색 (runtime error 방지를 위해 단순화)
        
        Args:
            entity_name: 검색할 엔티티 이름
            es_index: Elasticsearch 인덱스
            top_k: 반환할 결과 수
            
        Returns:
            List[텍스트 검색 결과들]
        """
        logger.info(f"📝 텍스트 검색 수행: {entity_name}")
        
        # 단순한 텍스트 검색 쿼리 (runtime error 방지)
        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        # 1) 완전 일치
                        {
                            "term": {
                                "concept_name.keyword": {
                                    "value": entity_name,
                                    "boost": 3.0
                                }
                            }
                        },
                        # 2) 부분 일치
                        {
                            "match": {
                                "concept_name": {
                                    "query": entity_name,
                                    "boost": 2.0
                                }
                            }
                        },
                        # 3) 구문 일치
                        {
                            "match_phrase": {
                                "concept_name": {
                                    "query": entity_name,
                                    "boost": 2.5
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            }
        }
        
        try:
            response = self.es_client.es_client.search(
                index=es_index,
                body=body
            )
            
            hits = response['hits']['hits'] if response['hits']['total']['value'] > 0 else []
            
            logger.info(f"✅ 텍스트 검색 완료: {len(hits)}개 결과")
            
            return hits
            
        except Exception as e:
            logger.error(f"텍스트 검색 실패: {e}")
            return []
    
    def _perform_text_only_search_silent(self, entity_name: str, es_index: str, top_k: int) -> List[Dict[str, Any]]:
        """
        간단한 텍스트 검색 (로깅 없는 버전)
        
        Args:
            entity_name: 검색할 엔티티 이름
            es_index: Elasticsearch 인덱스
            top_k: 반환할 결과 수
            
        Returns:
            List[텍스트 검색 결과들]
        """
        # 단순한 텍스트 검색 쿼리 (runtime error 방지)
        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        # 1) 완전 일치
                        {
                            "term": {
                                "concept_name.keyword": {
                                    "value": entity_name,
                                    "boost": 3.0
                                }
                            }
                        },
                        # 2) 부분 일치
                        {
                            "match": {
                                "concept_name": {
                                    "query": entity_name,
                                    "boost": 2.0
                                }
                            }
                        },
                        # 3) 구문 일치
                        {
                            "match_phrase": {
                                "concept_name": {
                                    "query": entity_name,
                                    "boost": 2.5
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            }
        }
        
        try:
            response = self.es_client.es_client.search(
                index=es_index,
                body=body
            )
            
            hits = response['hits']['hits'] if response['hits']['total']['value'] > 0 else []
            return hits
            
        except Exception as e:
            logger.error(f"텍스트 검색 실패: {e}")
            return []
    
    def _perform_native_hybrid_search(self, entity_name: str, entity_embedding: np.ndarray, es_index: str, top_k: int) -> List[Dict[str, Any]]:
        """
        네이티브 하이브리드 검색 (벡터 + 텍스트를 하나의 쿼리로 결합)
        
        Args:
            entity_name: 검색할 엔티티 이름
            entity_embedding: 엔티티 임베딩 벡터
            es_index: Elasticsearch 인덱스
            top_k: 반환할 결과 수
            
        Returns:
            List[하이브리드 검색 결과들]
        """
        logger.info(f"🔄 네이티브 하이브리드 검색 수행 (글자수 유사도 포함): {entity_name}")
        
        # 임베딩을 리스트로 변환
        embedding_list = entity_embedding.tolist()
        
        # 엔티티 이름 길이 계산
        entity_length = len(entity_name.strip())
        scale_len = max(8.0, entity_length * 0.8)
        
        # 하이브리드 쿼리 (knn + function_score로 글자수 유사도 추가)
        body = {
            "size": top_k,
            "knn": {
                "field": "concept_embedding",
                "query_vector": embedding_list,
                "k": top_k * 2,
                "num_candidates": top_k * 5,
                "boost": 0.5  # 벡터 검색 가중치 (글자수 고려로 조정)
            },
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "should": [
                                # 완전 일치
                                {
                                    "term": {
                                        "concept_name.keyword": {
                                            "value": entity_name,
                                            "boost": 3.0
                                        }
                                    }
                                },
                                # 구문 일치
                                {
                                    "match": {
                                        "concept_name": {
                                            "query": entity_name,
                                            "boost": 2.5
                                        }
                                    }
                                }
                            ],
                            "minimum_should_match": 1
                        }
                    },
                    # 글자수 유사도 함수
                    "functions": [
                        {
                            "script_score": {
                                "script": {
                                    "params": {
                                        "origin_len": float(entity_length),
                                        "scale_len": float(scale_len)
                                    },
                                    "source": """
                                        double origin = params.origin_len;
                                        double scale = params.scale_len;
                                        double len = 0.0;
                                        
                                        if (!doc['concept_name.keyword'].isEmpty()) {
                                            len = doc['concept_name.keyword'].value.length();
                                        } else if (!doc['concept_name'].isEmpty()) {
                                            len = doc['concept_name'].value.length();
                                        }
                                        
                                        // 가우시안 감쇠: exp(-0.5 * ((len-origin)/scale)^2)
                                        double x = (len - origin) / scale;
                                        double decay = Math.exp(-0.5 * x * x);
                                        
                                        // 길이 유사도 보너스 (1.0 ~ 2.0)
                                        return 1.0 + decay;
                                    """
                                }
                            }
                        }
                    ],
                    "score_mode": "multiply",  # 기본 점수와 길이 유사도 곱셈
                    "boost_mode": "multiply",
                    "boost": 0.3  # 텍스트 검색 가중치 (글자수 고려로 조정)
                }
            }
        }
        
        try:
            response = self.es_client.es_client.search(
                index=es_index,
                body=body
            )
            
            hits = response['hits']['hits'] if response['hits']['total']['value'] > 0 else []
            
            logger.info(f"✅ 네이티브 하이브리드 검색 완료: {len(hits)}개 결과")
            return hits
            
        except Exception as e:
            logger.error(f"네이티브 하이브리드 검색 실패: {e}")
            # 실패시 텍스트 검색만 수행
            logger.info("하이브리드 검색 실패 - 텍스트 검색으로 대체")
            return self._perform_text_only_search(entity_name, es_index, top_k)
    
    # def _apply_length_similarity_scoring(self, entity_name: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    #     """
    #     글자 길이 유사도를 고려한 점수 재조정
        
    #     Args:
    #         entity_name: 원본 엔티티 이름
    #         hits: 검색 결과들
            
    #     Returns:
    #         List[길이 유사도가 적용된 검색 결과들]
    #     """
    #     if not hits:
    #         return hits
        
    #     entity_length = len(entity_name.lower().strip())
    #     enhanced_hits = []
        
    #     for hit in hits:
    #         concept_name = hit['_source'].get('concept_name', '')
    #         concept_length = len(concept_name.lower().strip())
            
    #         # 길이 차이 계산
    #         length_diff = abs(entity_length - concept_length)
    #         max_length = max(entity_length, concept_length)
            
    #         # 길이 유사도 계산 (0.0 ~ 1.0)
    #         if max_length == 0:
    #             length_similarity = 1.0
    #         else:
    #             length_similarity = 1.0 - (length_diff / max_length)
            
    #         # 길이 유사도 가중치 적용
    #         # 길이가 비슷할수록 더 높은 점수
    #         length_weight = 0.15  # 15% 가중치
    #         original_score = hit['_score']
            
    #         # 길이 유사도 보너스/페널티 적용
    #         if length_similarity >= 0.9:  # 매우 유사한 길이
    #             length_bonus = 1.2
    #         elif length_similarity >= 0.8:  # 유사한 길이
    #             length_bonus = 1.1
    #         elif length_similarity >= 0.6:  # 보통 길이
    #             length_bonus = 1.0
    #         elif length_similarity >= 0.4:  # 다소 다른 길이
    #             length_bonus = 0.9
    #         else:  # 매우 다른 길이
    #             length_bonus = 0.8
            
    #         # 최종 점수 계산
    #         adjusted_score = original_score * (1 + length_weight * (length_bonus - 1))
            
    #         # 새로운 hit 객체 생성
    #         enhanced_hit = hit.copy()
    #         enhanced_hit['_score'] = adjusted_score
    #         enhanced_hit['_original_score'] = original_score
    #         enhanced_hit['length_similarity'] = length_similarity
    #         enhanced_hit['length_bonus'] = length_bonus
            
    #         enhanced_hits.append(enhanced_hit)
        
    #     # 조정된 점수로 재정렬
    #     enhanced_hits.sort(key=lambda x: x['_score'], reverse=True)
        
    #     return enhanced_hits
    
    def _get_standard_candidates(self, non_standard_concept_id: str, domain_id: str) -> List[Dict[str, Any]]:
        """
        Non-standard 컨셉의 Standard 후보들 조회
        concept_relationship 인덱스에서 "Maps to" 관계로 연결된 standard 컨셉들을 찾음
        
        Args:
            non_standard_concept_id: Non-standard 컨셉 ID
            domain_id: 도메인 ID
            
        Returns:
            List[Standard 컨셉 후보들]
        """
        try:
            standard_concept_ids = self._get_maps_to_relationships(non_standard_concept_id)
            standard_candidates = self._search_concepts_in_all_indices(standard_concept_ids, domain_id)
            
            logger.info(f"Non-standard {non_standard_concept_id}에 대한 {len(standard_candidates)}개 standard 후보 조회 완료")
            return standard_candidates
            
        except Exception as e:
            logger.error(f"Standard 후보 조회 오류: {str(e)}")
    
    def _get_maps_to_relationships(self, concept_id_1: str) -> List[str]:
        """
        concept-relationship 인덱스에서 Maps to 관계 조회
        
        Args:
            concept_id_1: 소스 컨셉 ID
            
        Returns:
            List[Maps to로 연결된 concept_id_2 리스트]
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
            
            # 디버깅 로그 추가
            logger.info(f"concept-relationship 인덱스에서 {concept_id_1}에 대한 {len(standard_concept_ids)}개 Maps to 관계 발견")
            if standard_concept_ids:
                logger.info(f"Maps to 관계로 찾은 concept_ids: {standard_concept_ids}")
            
            return standard_concept_ids
            
        except Exception as e:
            logger.warning(f"concept-relationship 인덱스 Maps to 관계 조회 실패: {str(e)}")
            return []
    
    def _search_concepts_in_all_indices(self, concept_ids: List[str], domain_id: str) -> List[Dict[str, Any]]:
        """
        엔티티 타입에 따라 지정된 도메인의 concept 인덱스에서 concept_id들 검색
        
        Args:
            concept_ids: 검색할 concept_id 리스트
            domain_id: 검색할 도메인 ID
            
        Returns:
            List[찾은 컨셉들]
        """
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
            
            concepts_response = self.es_client.es_client.search(
                index="concept",
                body=concepts_query
            )
            
            # 디버깅 로그 추가
            logger.info(f"검색 결과: {concepts_response['hits']['total']['value']}개 문서 발견")
            
            for hit in concepts_response['hits']['hits']:
                all_candidates.append(hit['_source'])
                
            if concepts_response['hits']['total']['value'] > 0:
                logger.info(f"{concepts_response['hits']['total']['value']}개 standard concept 발견")
            
        except Exception as e:
            logger.warning(f"검색 실패: {str(e)}")
        
        return all_candidates
    
    def _calculate_similarity(self, entity_name: str, concept_name: str) -> float:
        """
        두 문자열 간의 Jaccard 유사도 계산
        
        Args:
            entity_name: 원본 엔티티 이름
            concept_name: 비교할 컨셉 이름
            
        Returns:
            Jaccard 유사도 점수 (0.0 ~ 1.0)
        """
        if not entity_name or not concept_name:
            return 0.0
        
        # 대소문자 정규화
        entity_name = entity_name.lower()
        concept_name = concept_name.lower()
        
        # n-gram 3으로 분할
        entity_ngrams = self._get_ngrams(entity_name, n=3)
        concept_ngrams = self._get_ngrams(concept_name, n=3)
        
        if not entity_ngrams or not concept_ngrams:
            return 0.0
        
        # Jaccard 유사도 계산
        intersection = entity_ngrams.intersection(concept_ngrams)
        union = entity_ngrams.union(concept_ngrams)
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        
        return jaccard_similarity
    
    def _get_ngrams(self, text: str, n: int = 3) -> set:
        """
        텍스트를 n-gram으로 분할
        
        Args:
            text: 입력 텍스트
            n: n-gram 크기 (기본값: 3)
            
        Returns:
            n-gram 집합
        """
        if len(text) < n:
            return {text}
        
        ngrams = set()
        for i in range(len(text) - n + 1):
            ngrams.add(text[i:i + n])
        
        return ngrams
    
    def _create_mapping_result(self, entity_input: EntityInput, best_candidate: Dict[str, Any], alternative_candidates: List[Dict[str, Any]]) -> MappingResult:
        """
        매핑 결과 생성
        
        Args:
            entity_input: 원본 엔티티 입력
            best_candidate: 최적 후보
            alternative_candidates: 대안 후보들
            
        Returns:
            MappingResult: 매핑 결과
        """
        concept = best_candidate['concept']
        final_score = best_candidate['final_score']
        
        # 대안 컨셉들 추출
        alternative_concepts = []
        for alt_candidate in alternative_candidates:
            if 'concept' in alt_candidate:
                alt_concept = alt_candidate['concept']
                alternative_concepts.append({
                    'concept_id': str(alt_concept.get('concept_id', '')),
                    'concept_name': alt_concept.get('concept_name', ''),
                    'vocabulary_id': alt_concept.get('vocabulary_id', ''),
                    'score': alt_candidate.get('final_score', 0)
                })
        
        # 매핑 방법 결정
        mapping_method = "direct_standard" if best_candidate['is_original_standard'] else "non_standard_to_standard"
        
        # 매핑 신뢰도 계산 (final_score 사용)
        mapping_score = final_score
        mapping_confidence = self._determine_confidence(mapping_score)
        
        return MappingResult(
            source_entity=entity_input,
            mapped_concept_id=str(concept.get('concept_id', '')),
            mapped_concept_name=concept.get('concept_name', ''),
            domain_id=concept.get('domain_id', ''),
            vocabulary_id=concept.get('vocabulary_id', ''),
            concept_class_id=concept.get('concept_class_id', ''),
            standard_concept=concept.get('standard_concept', ''),
            concept_code=concept.get('concept_code', ''),
            valid_start_date=concept.get('valid_start_date'),
            valid_end_date=concept.get('valid_end_date'),
            invalid_reason=concept.get('invalid_reason'),
            concept_embedding=concept.get('concept_embedding'),
            mapping_score=mapping_score,
            mapping_confidence=mapping_confidence,
            mapping_method=mapping_method,
            alternative_concepts=alternative_concepts
        )
    
    def _determine_confidence(self, score: float) -> str:
        """
        매핑 신뢰도 결정 (0.0 ~ 1.0 점수 기준)
        
        신뢰도 기준:
        - 0.95 ~ 1.00: very_high (정확한 키워드 매칭)
        - 0.85 ~ 0.94: high (높은 유사도)
        - 0.70 ~ 0.84: medium (중간 유사도)
        - 0.50 ~ 0.69: low (낮은 유사도)
        - 0.00 ~ 0.49: very_low (매우 낮은 유사도)
        """
        if score >= 0.95:
            return "very_high"
        elif score >= 0.85:
            return "high"
        elif score >= 0.70:
            return "medium"
        elif score >= 0.50:
            return "low"
        else:
            return "very_low"
    
    def _calculate_hybrid_score(self, entity_name: str, concept_name: str, 
                              elasticsearch_score: float, concept_source: Dict[str, Any], 
                              text_weight: float = 0.4, semantic_weight: float = 0.6) -> tuple:
        """
        텍스트 유사도와 의미적 유사도를 결합한 하이브리드 점수 계산
        
        Args:
            entity_name: 엔티티 이름
            concept_name: 컨셉 이름
            elasticsearch_score: Elasticsearch 점수
            concept_source: 컨셉 소스 데이터
            text_weight: 텍스트 유사도 가중치 (기본값: 0.4)
            semantic_weight: 의미적 유사도 가중치 (기본값: 0.6)
            
        Returns:
            tuple: (하이브리드_점수, 텍스트_유사도, 의미적_유사도)
        """
        try:
            # 1. 텍스트 유사도 계산
            text_similarity = self._calculate_similarity(entity_name, concept_name)
            
            # 2. 의미적 유사도 계산
            concept_embedding = concept_source.get('concept_embedding')
            if concept_embedding and len(concept_embedding) == 768:
                # SapBERT 임베딩이 있는 경우 코사인 유사도 계산
                try:
                    # 엔티티 임베딩 생성 (SapBERT 사용)
                    entity_embedding = self._get_simple_embedding(entity_name) if HAS_SAPBERT else None
                    
                    if entity_embedding is not None:
                        concept_emb_array = np.array(concept_embedding).reshape(1, -1)
                        entity_emb_array = entity_embedding.reshape(1, -1)
                        semantic_similarity = cosine_similarity(entity_emb_array, concept_emb_array)[0][0]
                        # 코사인 유사도는 -1~1 범위이므로 0~1로 정규화
                        semantic_similarity = (semantic_similarity + 1.0) / 2.0
                        logger.debug(f"의미적 유사도 계산 성공: {semantic_similarity:.4f} for {concept_source.get('concept_name', 'N/A')}")
                    else:
                        # 임베딩 생성 실패시 0.0으로 설정
                        semantic_similarity = 0.0
                        logger.debug(f"엔티티 임베딩 생성 실패 - 의미적 유사도 0.0 사용: {concept_source.get('concept_name', 'N/A')}")
                        
                except Exception as e:
                    logger.warning(f"의미적 유사도 계산 실패: {e}")
                    semantic_similarity = 0.0
            else:
                # 임베딩이 없는 경우 0.0으로 설정 (텍스트 유사도와 구분)
                semantic_similarity = 0.0
                logger.debug(f"컨셉 임베딩 없음 - 의미적 유사도 0.0 사용: {concept_source.get('concept_name', 'N/A')}")
            
            # 3. 하이브리드 점수 계산
            hybrid_score = (text_weight * text_similarity) + (semantic_weight * semantic_similarity)
            
            # 점수를 0-1 범위로 제한
            hybrid_score = max(0.0, min(1.0, hybrid_score))
            text_similarity = max(0.0, min(1.0, text_similarity))
            semantic_similarity = max(0.0, min(1.0, semantic_similarity))
            
            return hybrid_score, text_similarity, semantic_similarity
            
        except Exception as e:
            logger.error(f"하이브리드 점수 계산 실패: {e}")
            # 오류 발생시 기본 Python 유사도 사용
            fallback_similarity = self._calculate_similarity(entity_name, concept_name)
            return fallback_similarity, fallback_similarity, fallback_similarity
    
    def _get_simple_embedding(self, text: str):
        """
        SapBERT를 사용하여 텍스트의 임베딩 생성
        """
        try:
            # SapBERT 모델이 초기화되어 있는지 확인
            if not hasattr(self, '_sapbert_model') or self._sapbert_model is None:
                self._initialize_sapbert_model()
            
            if self._sapbert_model is None:
                return None
            
            # 텍스트를 소문자로 변환
            text = text.lower().strip()
            
            # 토크나이징
            inputs = self._sapbert_tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=25
            )
            inputs = {k: v.to(self._sapbert_device) for k, v in inputs.items()}
            
            # 임베딩 생성
            with torch.no_grad():
                outputs = self._sapbert_model(**inputs)
                # CLS 토큰의 임베딩 사용
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
            return embedding.flatten()
            
        except Exception as e:
            logger.warning(f"SapBERT 임베딩 생성 실패: {e}")
            return None
    
    def _initialize_sapbert_model(self):
        """SapBERT 모델 초기화 (지연 로딩)"""
        try:
            if not HAS_SAPBERT:
                logger.warning("SapBERT 관련 패키지가 설치되지 않음")
                self._sapbert_model = None
                self._sapbert_tokenizer = None
                self._sapbert_device = None
                return
            
            model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
            logger.info(f"🤖 SapBERT 모델 로딩 중: {model_name}")
            
            self._sapbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._sapbert_model = AutoModel.from_pretrained(model_name)
            # GPU가 다른 작업으로 점유된 경우 CPU 사용
            self._sapbert_device = torch.device('cpu')  # 임시로 CPU 강제 사용
            self._sapbert_model.to(self._sapbert_device)
            self._sapbert_model.eval()
            
            logger.info(f"✅ SapBERT 모델 로딩 완료 (Device: {self._sapbert_device})")
            
        except Exception as e:
            logger.error(f"❌ SapBERT 모델 로딩 실패: {e}")
            self._sapbert_model = None
            self._sapbert_tokenizer = None
            self._sapbert_device = None
    
    def health_check(self) -> Dict[str, Any]:
        """API 상태 확인"""
        es_health = self.es_client.health_check()
        
        return {
            "api_status": "healthy",
            "elasticsearch_status": es_health,
            "confidence_threshold": self.confidence_threshold
        }

# API 편의 함수들
def map_single_entity(
    entity_name: str,
    entity_type: str,
    domain_id: Optional[DomainID] = None,
    vocabulary_id: Optional[str] = None,
    confidence: float = 1.0
) -> Optional[MappingResult]:
    """
    단일 엔티티 매핑 편의 함수
    
    Args:
        entity_name: 엔티티 이름
        entity_type: 엔티티 타입 ('diagnostic', 'drug', 'test', 'surgery')
        domain_id: OMOP 도메인 ID (선택사항)
        vocabulary_id: OMOP 어휘체계 ID (선택사항)
        confidence: 엔티티 신뢰도
        
    Returns:
        MappingResult: 매핑 결과 또는 None
    """
    try:
        api = EntityMappingAPI()
        
        entity_input = EntityInput(
            entity_name=entity_name,
            domain_id=domain_id if isinstance(domain_id, DomainID) else (DomainID(domain_id) if domain_id else None),
            vocabulary_id=vocabulary_id,
            confidence=confidence
        )
        
        return api.map_entity(entity_input)
        
    except ValueError:
        logger.error(f"지원하지 않는 엔티티 타입: {entity_type}")
        return None