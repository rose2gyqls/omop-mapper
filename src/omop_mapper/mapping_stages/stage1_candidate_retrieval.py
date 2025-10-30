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
    """Stage 1: 후보군 추출 (Lexical 5 + Semantic 5 + Combined 5)"""
    
    def __init__(self, es_client, has_sapbert: bool = True):
        """
        Args:
            es_client: Elasticsearch 클라이언트
            has_sapbert: SapBERT 사용 가능 여부
        """
        self.es_client = es_client
        self.has_sapbert = has_sapbert
    
    def retrieve_candidates(
        self, 
        entity_name: str, 
        domain_id: str,
        entity_embedding: Optional[np.ndarray] = None,
        es_index: str = "concept-small"
    ) -> List[Dict[str, Any]]:
        """
        각 도메인별로 lexical 5개, semantic 5개, combined 5개 후보 추출 (총 15개)
        
        Args:
            entity_name: 엔티티 이름
            domain_id: 도메인 ID (필터링에 사용)
            entity_embedding: 엔티티 임베딩 벡터 (선택사항)
            es_index: Elasticsearch 인덱스
            
        Returns:
            List[Dict]: 15개의 후보 리스트 (각 후보는 검색 타입 정보 포함)
        """
        logger.info("=" * 80)
        logger.info("Stage 1: 각 도메인별 후보군 15개 추출")
        logger.info("  - Lexical Analysis: 5개")
        logger.info("  - Semantic Analysis: 5개")
        logger.info("  - Combined Score: 5개")
        logger.info("=" * 80)
        
        logger.info(f"🔍 엔티티: {entity_name}")
        logger.info(f"🔍 도메인: {domain_id}")
        
        all_candidates = []
        
        # 1. Lexical Analysis - 텍스트 검색으로 top 5개
        logger.info("\n" + "=" * 60)
        logger.info("📝 1-1. Lexical Analysis (텍스트 검색)")
        logger.info("=" * 60)
        lexical_results = self._perform_text_only_search(entity_name, domain_id, es_index, 5)
        logger.info(f"✅ Lexical 후보: {len(lexical_results)}개")
        for i, hit in enumerate(lexical_results, 1):
            source = hit['_source']
            hit['_search_type'] = 'lexical'
            all_candidates.append(hit)
            logger.info(f"  {i}. {source.get('concept_name', 'N/A')} "
                      f"(ID: {source.get('concept_id', 'N/A')}) "
                      f"[Domain: {source.get('domain_id', 'N/A')}] "
                      f"- 점수: {hit['_score']:.4f}")
        
        # 2. Semantic Analysis - 벡터 검색으로 top 5개
        logger.info("\n" + "=" * 60)
        logger.info("🧠 1-2. Semantic Analysis (벡터 검색)")
        logger.info("=" * 60)
        if entity_embedding is not None:
            semantic_results = self._perform_vector_search(entity_embedding, domain_id, es_index, 5)
            logger.info(f"✅ Semantic 후보: {len(semantic_results)}개")
            for i, hit in enumerate(semantic_results, 1):
                source = hit['_source']
                hit['_search_type'] = 'semantic'
                all_candidates.append(hit)
                logger.info(f"  {i}. {source.get('concept_name', 'N/A')} "
                          f"(ID: {source.get('concept_id', 'N/A')}) "
                          f"[Domain: {source.get('domain_id', 'N/A')}] "
                          f"- 점수: {hit['_score']:.4f}")
        else:
            logger.warning("⚠️ 임베딩 없음 - Semantic 검색 건너뜀")
        
        # 3. Combined Score - 하이브리드 검색으로 top 5개
        logger.info("\n" + "=" * 60)
        logger.info("🔄 1-3. Combined Score (하이브리드 검색)")
        logger.info("=" * 60)
        if entity_embedding is not None:
            combined_results = self._perform_native_hybrid_search(
                entity_name, entity_embedding, domain_id, es_index, 5
            )
        else:
            # 임베딩이 없으면 텍스트 검색 결과 재사용
            combined_results = lexical_results[:5]
        
        logger.info(f"✅ Combined 후보: {len(combined_results)}개")
        for i, hit in enumerate(combined_results, 1):
            source = hit['_source']
            hit['_search_type'] = 'combined'
            all_candidates.append(hit)
            logger.info(f"  {i}. {source.get('concept_name', 'N/A')} "
                      f"(ID: {source.get('concept_id', 'N/A')}) "
                      f"[Domain: {source.get('domain_id', 'N/A')}] "
                      f"- 점수: {hit['_score']:.4f}")
        
        # 최종 요약
        logger.info("\n" + "=" * 80)
        logger.info(f"📊 Stage 1 완료: 총 {len(all_candidates)}개 후보 추출")
        logger.info(f"  - Lexical: {len(lexical_results)}개")
        logger.info(f"  - Semantic: {len(semantic_results) if entity_embedding is not None else 0}개")
        logger.info(f"  - Combined: {len(combined_results)}개")
        logger.info("=" * 80)
        
        return all_candidates
    
    def _perform_text_only_search(self, entity_name: str, domain_id: str, es_index: str, top_k: int) -> List[Dict[str, Any]]:
        """텍스트 검색 수행 (domain_id 필터 적용)"""
        # Measurement 도메인의 경우 Meas Value도 포함
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
        """벡터 검색 수행 (domain_id 필터 적용)"""
        embedding_list = entity_embedding.tolist()
        
        # Measurement 도메인의 경우 Meas Value도 포함
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
        """네이티브 하이브리드 검색 (벡터 + 텍스트 + 글자수 유사도 + domain_id 필터)"""
        embedding_list = entity_embedding.tolist()
        entity_length = len(entity_name.strip())
        scale_len = max(8.0, entity_length * 0.8)
        
        # Measurement 도메인의 경우 Meas Value도 포함
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

