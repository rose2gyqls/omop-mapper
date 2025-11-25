"""
Stage 1: Elasticsearch에서 각 도메인별 후보군 15개 추출
- Lexical Analysis: 텍스트 기반 검색으로 top 5개
- Semantic Analysis: 의미적 검색으로 top 5개
- Combined Score: 하이브리드 검색으로 top 5개
"""
from typing import List, Dict, Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Stage1CandidateRetrieval:
    """Stage 1: 후보군 추출 (Lexical 3 + Semantic 3 + Combined 3)"""
    
    def __init__(self, es_client, has_sapbert: bool = True):
        """
        Args:
            es_client: Elasticsearch 클라이언트
            has_sapbert: SapBERT 사용 가능 여부
        """
        self.es_client = es_client
        self.has_sapbert = has_sapbert
        # Threshold 설정
        self.lexical_threshold = 5.0
        self.semantic_threshold = 0.8
        self.combined_threshold = 5.0
    
    def retrieve_candidates(
        self, 
        entity_name: str, 
        domain_id: str,
        entity_embedding: Optional[np.ndarray] = None,
        es_index: str = "concept-small"
    ) -> List[Dict[str, Any]]:
        """
        각 도메인별로 다양한 검색 전략을 사용하여 후보군 추출
        
        **검색 전략**:
        - Lexical: 텍스트 기반 검색 (최대 3개)
        - Semantic: 벡터 기반 검색 (최대 3개, 임베딩이 있는 경우)
        - Combined: 하이브리드 검색 (최대 3개, 임베딩이 있는 경우)
        
        **Threshold 필터링**:
        - Lexical: {self.lexical_threshold} 이상
        - Semantic: {self.semantic_threshold} 이상
        - Combined: {self.combined_threshold} 이상
        
        Args:
            entity_name: 엔티티 이름
            domain_id: 검색할 도메인 ID (해당 도메인만 필터링)
            entity_embedding: 엔티티 임베딩 벡터 (선택사항)
            es_index: Elasticsearch 인덱스 이름
            
        Returns:
            List[Dict]: Threshold를 통과한 후보 리스트 (각 후보는 _search_type 필드 포함)
        """
        logger.info("=" * 80)
        logger.info("Stage 1: 후보군 추출 (Lexical + Semantic + Combined)")
        logger.info(f"  엔티티: {entity_name}")
        logger.info(f"  도메인: {domain_id}")
        logger.info("=" * 80)
        
        all_candidates = []
        
        # ===== 1. Lexical Analysis: 텍스트 기반 검색 =====
        logger.info("\n📝 1-1. Lexical Analysis (텍스트 검색, threshold: {:.2f})".format(self.lexical_threshold))
        lexical_results = self._perform_text_only_search(entity_name, domain_id, es_index, 3)
        lexical_results_filtered = [hit for hit in lexical_results if hit['_score'] >= self.lexical_threshold]
        logger.info(f"✅ Lexical: {len(lexical_results)}개 → {len(lexical_results_filtered)}개 (threshold 통과)")
        
        for hit in lexical_results_filtered:
            hit['_search_type'] = 'lexical'
            all_candidates.append(hit)
            source = hit['_source']
            logger.debug(f"  - {source.get('concept_name', 'N/A')} (ID: {source.get('concept_id', 'N/A')}) [점수: {hit['_score']:.4f}]")
        
        # ===== 2. Semantic Analysis: 벡터 기반 검색 =====
        logger.info("\n🧠 1-2. Semantic Analysis (벡터 검색, threshold: {:.2f})".format(self.semantic_threshold))
        semantic_results_filtered = []
        if entity_embedding is not None:
            semantic_results = self._perform_vector_search(entity_embedding, domain_id, es_index, 3)
            semantic_results_filtered = [hit for hit in semantic_results if hit['_score'] >= self.semantic_threshold]
            logger.info(f"✅ Semantic: {len(semantic_results)}개 → {len(semantic_results_filtered)}개 (threshold 통과)")
            
            for hit in semantic_results_filtered:
                hit['_search_type'] = 'semantic'
                all_candidates.append(hit)
                source = hit['_source']
                logger.debug(f"  - {source.get('concept_name', 'N/A')} (ID: {source.get('concept_id', 'N/A')}) [점수: {hit['_score']:.4f}]")
        else:
            logger.warning("⚠️ 임베딩 없음 - Semantic 검색 건너뜀")
        
        # ===== 3. Combined Score: 하이브리드 검색 =====
        logger.info("\n🔄 1-3. Combined Score (하이브리드 검색, threshold: {:.2f})".format(self.combined_threshold))
        combined_results_filtered = []
        if entity_embedding is not None:
            combined_results = self._perform_native_hybrid_search(
                entity_name, entity_embedding, domain_id, es_index, 3
            )
            combined_results_filtered = [hit for hit in combined_results if hit['_score'] >= self.combined_threshold]
        else:
            # 임베딩이 없으면 텍스트 검색 결과 재사용
            combined_results = lexical_results[:3]
            combined_results_filtered = [hit for hit in combined_results if hit['_score'] >= self.combined_threshold]
        
        logger.info(f"✅ Combined: {len(combined_results if entity_embedding is not None else lexical_results[:3])}개 → {len(combined_results_filtered)}개 (threshold 통과)")
        for hit in combined_results_filtered:
            hit['_search_type'] = 'combined'
            all_candidates.append(hit)
            source = hit['_source']
            logger.debug(f"  - {source.get('concept_name', 'N/A')} (ID: {source.get('concept_id', 'N/A')}) [점수: {hit['_score']:.4f}]")
        
        # 최종 요약
        logger.info("\n" + "=" * 80)
        logger.info(f"📊 Stage 1 완료: 총 {len(all_candidates)}개 후보 추출")
        logger.info(f"  - Lexical: {len(lexical_results_filtered)}개 (threshold: {self.lexical_threshold:.2f})")
        logger.info(f"  - Semantic: {len(semantic_results_filtered)}개 (threshold: {self.semantic_threshold:.2f})")
        logger.info(f"  - Combined: {len(combined_results_filtered)}개 (threshold: {self.combined_threshold:.2f})")
        logger.info("=" * 80)
        
        return all_candidates

    def _perform_text_only_search(self, entity_name: str, domain_id: str, es_index: str, top_k: int) -> List[Dict[str, Any]]:
        """
        텍스트 기반 검색 수행 (Lexical Search)
        
        **검색 전략**:
        - Exact match: concept_name.keyword로 정확히 일치하는 항목 (boost: 3.0)
        - Phrase match: concept_name에 구문 일치하는 항목 (boost: 2.5)
        - Text match: concept_name에 텍스트 일치하는 항목 (boost: 2.0)
        
        Args:
            entity_name: 검색할 엔티티 이름
            domain_id: 도메인 필터 (해당 도메인만 검색)
            es_index: Elasticsearch 인덱스
            top_k: 반환할 최대 결과 수
            
        Returns:
            List[Dict]: 검색 결과 리스트
        """
        # Measurement 도메인의 경우 "Meas Value"도 포함 (OMOP CDM 특성)
        if domain_id == "Measurement":
            domain_filter = {
                "terms": {
                    "domain_id": ["Measurement", "Meas Value"]
                }
            }
        else:
            domain_filter = {
                "term": {
                    "domain_id": domain_id
                }
            }
        
        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "bool": {
                                "should": [
                                    {
                                        "term": {
                                            "concept_name.keyword": {
                                                "value": entity_name,
                                                "boost": 3.0
                                            }
                                        }
                                    },
                                    {
                                        "match_phrase": {
                                            "concept_name": {
                                                "query": entity_name,
                                                "boost": 2.5
                                            }
                                        }
                                    },
                                    {
                                        "match": {
                                            "concept_name": {
                                                "query": entity_name,
                                                "boost": 2.0
                                            }
                                        }
                                    }
                                ],
                                "minimum_should_match": 1
                            }
                        }
                    ],
                    "filter": [
                        domain_filter
                    ]
                }
            }
        }
        
        try:
            response = self.es_client.es_client.search(index=es_index, body=body)
            hits = response['hits']['hits'] if response['hits']['total']['value'] > 0 else []
            return hits
        except Exception as e:
            logger.error(f"텍스트 검색 실패: {e}")
            return []
    
    def _perform_vector_search(self, entity_embedding: np.ndarray, domain_id: str, es_index: str, top_k: int) -> List[Dict[str, Any]]:
        """
        벡터 기반 검색 수행 (Semantic Search)
        
        **검색 전략**:
        - Elasticsearch KNN (k-Nearest Neighbors) 검색 사용
        - concept_embedding 필드와 입력 임베딩 간의 유사도 계산
        - 코사인 유사도 기반으로 가장 유사한 개념 검색
        
        Args:
            entity_embedding: 엔티티의 임베딩 벡터 (SapBERT 등)
            domain_id: 도메인 필터 (해당 도메인만 검색)
            es_index: Elasticsearch 인덱스
            top_k: 반환할 최대 결과 수
            
        Returns:
            List[Dict]: 검색 결과 리스트
        """
        embedding_list = entity_embedding.tolist()
        
        # Measurement 도메인의 경우 "Meas Value"도 포함 (OMOP CDM 특성)
        if domain_id == "Measurement":
            domain_filter = {
                "terms": {
                    "domain_id": ["Measurement", "Meas Value"]
                }
            }
        else:
            domain_filter = {
                "term": {
                    "domain_id": domain_id
                }
            }
        
        vector_query = {
            "knn": {
                "field": "concept_embedding",
                "query_vector": embedding_list,
                "k": top_k,
                "num_candidates": top_k * 3,
                "filter": domain_filter
            },
            "size": top_k,
            "_source": True
        }
        
        try:
            response = self.es_client.es_client.search(index=es_index, body=vector_query)
            hits = response['hits']['hits'] if response['hits']['total']['value'] > 0 else []
            return hits
        except Exception as e:
            logger.error(f"벡터 검색 실패: {e}")
            return []
    
    def _perform_native_hybrid_search(
        self, 
        entity_name: str, 
        entity_embedding: np.ndarray,
        domain_id: str,
        es_index: str, 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        하이브리드 검색 수행 (텍스트 + 벡터 + 길이 유사도)
        
        **검색 전략**:
        - KNN 벡터 검색 (boost: 0.6)
        - 텍스트 검색 (exact match boost: 3.0, match boost: 2.5)
        - 길이 유사도 가중치 (가우시안 decay 함수 사용)
        
        **길이 유사도**:
        - 입력 엔티티와 후보 개념의 글자 수 차이를 고려
        - 유사한 길이의 개념에 높은 가중치 부여
        
        Args:
            entity_name: 검색할 엔티티 이름
            entity_embedding: 엔티티 임베딩 벡터
            domain_id: 도메인 필터
            es_index: Elasticsearch 인덱스
            top_k: 반환할 최대 결과 수
            
        Returns:
            List[Dict]: 검색 결과 리스트
        """
        embedding_list = entity_embedding.tolist()
        entity_length = len(entity_name.strip())
        scale_len = max(8.0, entity_length * 0.8)
        
        # Measurement 도메인의 경우 "Meas Value"도 포함 (OMOP CDM 특성)
        if domain_id == "Measurement":
            domain_filter = {
                "terms": {
                    "domain_id": ["Measurement", "Meas Value"]
                }
            }
        else:
            domain_filter = {
                "term": {
                    "domain_id": domain_id
                }
            }
        
        body = {
            "size": top_k,
            "knn": {
                "field": "concept_embedding",
                "query_vector": embedding_list,
                "k": top_k * 2,
                "num_candidates": top_k * 5,
                "boost": 0.6,
                "filter": domain_filter
            },
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "bool": {
                                        "should": [
                                            {
                                                "term": {
                                                    "concept_name.keyword": {
                                                        "value": entity_name,
                                                        "boost": 3.0
                                                    }
                                                }
                                            },
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
                                }
                            ],
                            "filter": [
                                domain_filter
                            ]
                        }
                    },
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
                                        
                                        double x = (len - origin) / scale;
                                        double decay = Math.exp(-0.5 * x * x);
                                        
                                        return 1.0 + decay;
                                    """
                                }
                            }
                        }
                    ],
                    "score_mode": "multiply",
                    "boost_mode": "multiply",
                    "boost": 0.4
                }
            }
        }
        
        try:
            response = self.es_client.es_client.search(index=es_index, body=body)
            hits = response['hits']['hits'] if response['hits']['total']['value'] > 0 else []
            return hits
        except Exception as e:
            logger.error(f"하이브리드 검색 실패: {e}")
            # 실패시 텍스트 검색으로 대체
            return self._perform_text_only_search(entity_name, domain_id, es_index, top_k)

