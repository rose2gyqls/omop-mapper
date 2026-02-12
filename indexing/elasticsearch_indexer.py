"""
Elasticsearch Indexer Module

Provides functionality to index OMOP CDM data into Elasticsearch.
Supports CONCEPT, CONCEPT_RELATIONSHIP, and CONCEPT_SYNONYM tables.

Robust indexing:
    - 429 Too Many Requests → 지수 백오프 재시도 (5~300초, 최대 7회)
    - Bulk 응답 내 개별 실패 문서 → 자동 재시도 (최대 3회)
    - Idempotent _id 기반 → 재전송 시 덮어쓰기 (중복 없음)
"""

import logging
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


class ElasticsearchIndexer:
    """Elasticsearch indexer for OMOP CDM data."""
    
    # 429 재시도 설정
    MAX_429_RETRIES = 7          # 429 최대 재시도 횟수
    INITIAL_BACKOFF_SEC = 5      # 첫 백오프 대기 시간 (5, 10, 20, 40, 80, 160, 300초)
    MAX_BACKOFF_SEC = 300        # 최대 백오프 대기 시간 (5분)
    MAX_ITEM_RETRIES = 3         # 개별 실패 문서 최대 재시도 횟수
    
    # Index mapping configurations
    MAPPINGS = {
        'concept': {
                    "mappings": {
                        "properties": {
                            "concept_id": {"type": "keyword"},
                            "concept_name": {
                                "type": "text",
                                "similarity": "custom_bm25",
                                "fields": {
                                    "keyword": {"type": "keyword"},
                                    "trigram": {
                                        "type": "text",
                                        "analyzer": "trigram_analyzer",
                                        "similarity": "custom_bm25"
                                    }
                                }
                            },
                            "domain_id": {"type": "keyword"},
                            "vocabulary_id": {"type": "keyword"},
                            "concept_class_id": {"type": "keyword"},
                            "standard_concept": {"type": "keyword"},
                            "concept_code": {"type": "keyword"},
                            "valid_start_date": {"type": "date", "format": "yyyyMMdd"},
                            "valid_end_date": {"type": "date", "format": "yyyyMMdd"},
                            "invalid_reason": {"type": "keyword"}
                }
            }
        },
        'concept-small': {
            "mappings": {
                "properties": {
                    "concept_id": {"type": "keyword"},
                    "concept_name": {
                        "type": "text",
                        "similarity": "custom_bm25",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "trigram": {
                                "type": "text",
                                "analyzer": "trigram_analyzer",
                                "similarity": "custom_bm25"
                            }
                        }
                    },
                    "name_type": {"type": "keyword"},
                    "domain_id": {"type": "keyword"},
                    "vocabulary_id": {"type": "keyword"},
                    "concept_class_id": {"type": "keyword"},
                    "standard_concept": {"type": "keyword"},
                    "concept_code": {"type": "keyword"},
                    "valid_start_date": {"type": "date", "format": "yyyyMMdd"},
                    "valid_end_date": {"type": "date", "format": "yyyyMMdd"},
                    "invalid_reason": {"type": "keyword"}
                }
            }
        },
        'concept-relationship': {
            "mappings": {
                "properties": {
                    "concept_id_1": {"type": "keyword"},
                    "concept_id_2": {"type": "keyword"},
                    "relationship_id": {"type": "keyword"},
                    "valid_start_date": {"type": "date", "format": "yyyyMMdd"},
                    "valid_end_date": {"type": "date", "format": "yyyyMMdd"},
                    "invalid_reason": {"type": "keyword"}
                }
            }
        },
        'concept-synonym': {
            "mappings": {
                "properties": {
                    "concept_id": {"type": "keyword"},
                    "concept_synonym_name": {"type": "keyword"},
                    "language_concept_id": {"type": "keyword"}
                }
            }
        }
    }
    
    # Common index settings
    INDEX_SETTINGS = {
                        "number_of_shards": 3,
                        "number_of_replicas": 5,
                        "refresh_interval": "30s",
                        "index.write.wait_for_active_shards": "1",
                        "index.max_result_window": 50000,
                        "similarity": {
                            "custom_bm25": {
                                "type": "BM25",
                                "k1": 0.9,
                                "b": 0.5
                            }
                        },
                        "analysis": {
                            "analyzer": {
                                "trigram_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "standard",
                                    "filter": ["lowercase", "trigram_filter"]
                                }
                            },
                            "filter": {
                                "trigram_filter": {
                                    "type": "ngram",
                                    "min_gram": 4,
                                    "max_gram": 5
                                }
                            }
                        }
                    }
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        scheme: str = "http",
        username: Optional[str] = None,
        password: Optional[str] = None,
        index_name: str = "concept",
        include_embeddings: bool = True
    ):
        """
        Initialize Elasticsearch indexer.
        
        Args:
            host: Elasticsearch host
            port: Elasticsearch port
            scheme: Connection scheme (http/https)
            username: Username for authentication
            password: Password for authentication
            index_name: Target index name
            include_embeddings: Whether to include vector embeddings
        """
        self.index_name = index_name
        self.include_embeddings = include_embeddings
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create Elasticsearch client
        self.es = self._create_client(host, port, scheme, username, password)
        
        # Test connection (with 429 retry)
        self._ping_with_retry()
        
        self.logger.info(f"Connected to Elasticsearch at {host}:{port}")
    
    def _create_client(
        self,
        host: str,
        port: int,
        scheme: str,
        username: Optional[str],
        password: Optional[str]
    ) -> Elasticsearch:
        """Create Elasticsearch client with retry logic."""
        try:
            if username and password:
                return Elasticsearch(
                    hosts=[{"host": host, "port": port, "scheme": scheme}],
                    basic_auth=(username, password),
                    request_timeout=120,
                    retry_on_timeout=True,
                    max_retries=3
                )
            else:
                return Elasticsearch(
                    hosts=[{"host": host, "port": port, "scheme": scheme}],
                    request_timeout=120,
                    retry_on_timeout=True,
                    max_retries=3
                )
        except Exception as e:
            # Fallback for older elasticsearch-py versions
            try:
                return Elasticsearch([f"{scheme}://{host}:{port}"])
            except Exception as e2:
                raise ConnectionError(f"Failed to create ES client: {e}, fallback failed: {e2}")
    
    def _ping_with_retry(self):
        """Ping Elasticsearch with 429 retry."""
        for attempt in range(self.MAX_429_RETRIES + 1):
            try:
                if self.es.ping():
                    return
                raise ConnectionError("Elasticsearch ping failed")
            except Exception as e:
                status = getattr(e, 'status_code', None)
                if status == 429 and attempt < self.MAX_429_RETRIES:
                    wait = min(self.INITIAL_BACKOFF_SEC * (2 ** attempt), self.MAX_BACKOFF_SEC)
                    self.logger.warning(f"429 on ping, waiting {wait}s (retry {attempt+1}/{self.MAX_429_RETRIES})")
                    time.sleep(wait)
                else:
                    raise
    
    def _get_index_mapping(self) -> Dict:
        """Get appropriate mapping for the index type."""
        # Determine index type from name
        index_lower = self.index_name.lower()
        
        if "relationship" in index_lower:
            mapping = self.MAPPINGS['concept-relationship'].copy()
        elif "synonym" in index_lower:
            mapping = self.MAPPINGS['concept-synonym'].copy()
        elif "small" in index_lower:
            mapping = self.MAPPINGS['concept-small'].copy()
        else:
            mapping = self.MAPPINGS['concept'].copy()
        
        # Deep copy mappings to avoid mutation
        import copy
        mapping = copy.deepcopy(mapping)
        
        # Add settings
        mapping["settings"] = self.INDEX_SETTINGS.copy()
        
        # Add embedding field if enabled (concept, concept-small만 해당)
        # synonym, relationship은 임베딩 불필요
        if self.include_embeddings:
            if "synonym" not in index_lower and "relationship" not in index_lower:
                embedding_field = {
                    "type": "dense_vector",
                    "dims": 768,
                    "index": True,
                    "similarity": "cosine"
                }
                mapping["mappings"]["properties"]["concept_embedding"] = embedding_field
        
        return mapping
    
    def create_index(self, delete_if_exists: bool = False) -> bool:
        """
        Create Elasticsearch index with 429 retry.
        
        Args:
            delete_if_exists: Delete existing index if it exists
            
        Returns:
            True if index was created successfully
        """
        for attempt in range(self.MAX_429_RETRIES + 1):
            try:
                # Check if index exists
                if self.es.indices.exists(index=self.index_name):
                    if delete_if_exists:
                        self.logger.info(f"Deleting existing index: {self.index_name}")
                        self.es.indices.delete(index=self.index_name)
                    else:
                        self.logger.info(f"Index already exists: {self.index_name}")
                        return True
                
                # Create index with mapping
                mapping = self._get_index_mapping()
                self.es.indices.create(index=self.index_name, body=mapping)
                self.logger.info(f"Index created: {self.index_name}")
                
                # Wait for index to be ready
                time.sleep(2)
                
                try:
                    health = self.es.cluster.health(
                        index=self.index_name, 
                        wait_for_status="yellow", 
                        timeout="10s"
                    )
                    self.logger.info(f"Index health: {health['status']}")
                except Exception as e:
                    self.logger.warning(f"Could not verify index health: {e}")
                
                return True
                
            except Exception as e:
                status = getattr(e, 'status_code', None)
                if status == 429 and attempt < self.MAX_429_RETRIES:
                    wait = min(self.INITIAL_BACKOFF_SEC * (2 ** attempt), self.MAX_BACKOFF_SEC)
                    self.logger.warning(
                        f"429 on create_index, waiting {wait}s "
                        f"(retry {attempt+1}/{self.MAX_429_RETRIES})"
                    )
                    time.sleep(wait)
                else:
                    self.logger.error(f"Failed to create index: {e}")
                    return False
        
        return False
    
    def _generate_doc_id(self, doc: Dict) -> str:
        """Generate deterministic document ID based on index type (멱등성 보장)."""
        index_lower = self.index_name.lower()
        
        if "relationship" in index_lower:
            unique_str = f"{doc.get('concept_id_1', '')}_{doc.get('concept_id_2', '')}_{doc.get('relationship_id', '')}"
            return hashlib.md5(unique_str.encode()).hexdigest()
        elif "synonym" in index_lower:
            unique_str = f"{doc.get('concept_id', '')}_{doc.get('concept_synonym_name', '')}"
            return hashlib.md5(unique_str.encode()).hexdigest()
        elif "small" in index_lower:
            # concept-small: concept_id + concept_name으로 고유 ID 생성
            unique_str = f"{doc.get('concept_id', '')}_{doc.get('concept_name', '')}"
            return hashlib.md5(unique_str.encode()).hexdigest()
        else:
            return str(doc.get("concept_id", ""))
    
    def _prepare_actions(self, documents: List[Dict[str, Any]]) -> List[Dict]:
        """Prepare bulk actions from documents."""
        actions = []
        for doc in documents:
            if not self.include_embeddings:
                doc = doc.copy()
                doc.pop("concept_embedding", None)
                doc.pop("concept_synonym_embedding", None)
            
            actions.append({
                "_index": self.index_name,
                "_id": self._generate_doc_id(doc),
                "_source": doc
            })
        return actions
    
    def _send_bulk_with_retry(
        self,
        actions: List[Dict],
    ) -> Tuple[int, int]:
        """
        Send bulk request with robust retry logic.
        
        1. 429 → 지수 백오프로 전체 배치 재시도 (5, 10, 20, 40, 80, 160, 300초)
        2. 응답 내 개별 실패 문서 → 실패한 문서만 재시도 (최대 3회)
        
        Returns:
            (success_count, failed_count)
        """
        for attempt in range(self.MAX_429_RETRIES + 1):
            try:
                success, failed_items = bulk(
                    client=self.es,
                    actions=actions,
                    chunk_size=min(len(actions), 500),
                    request_timeout=600,
                    raise_on_error=False,
                    raise_on_exception=True,
                    max_retries=0,
                    refresh=False
                )
                
                if not failed_items:
                    return (success, 0)
                
                # ── 개별 실패 문서 재시도 ──
                self.logger.warning(f"{len(failed_items)} items failed in bulk response")
                
                # 실패한 _id 추출
                failed_ids = set()
                for item in failed_items:
                    for action_type, info in item.items():
                        fid = info.get('_id', '')
                        failed_ids.add(fid)
                        err = info.get('error', {})
                        self.logger.debug(
                            f"Failed: _id={fid}, status={info.get('status')}, "
                            f"error={err.get('type', 'unknown')}: {err.get('reason', '')}"
                        )
                
                retry_actions = [a for a in actions if a.get('_id') in failed_ids]
                recovered = 0
                
                for item_attempt in range(self.MAX_ITEM_RETRIES):
                    if not retry_actions:
                        break
                    
                    wait = min(2 * (2 ** item_attempt), 30)  # 2, 4, 8초 (최대 30초)
                    time.sleep(wait)
                    self.logger.info(
                        f"Retrying {len(retry_actions)} failed items "
                        f"(attempt {item_attempt+1}/{self.MAX_ITEM_RETRIES})"
                    )
                    
                    try:
                        r_success, r_failed = bulk(
                            client=self.es,
                            actions=retry_actions,
                            chunk_size=min(len(retry_actions), 200),
                            request_timeout=600,
                            raise_on_error=False,
                            raise_on_exception=True,
                            max_retries=0,
                            refresh=False
                        )
                        recovered += r_success
                        
                        if not r_failed:
                            retry_actions = []
                            break
                        
                        # 여전히 실패한 것만 다시 추출
                        still_failed_ids = set()
                        for item in r_failed:
                            for at, info in item.items():
                                still_failed_ids.add(info.get('_id', ''))
                        retry_actions = [a for a in retry_actions if a.get('_id') in still_failed_ids]
                        
                    except Exception as re:
                        self.logger.warning(f"Item retry error: {re}")
                        if getattr(re, 'status_code', None) == 429:
                            time.sleep(30)
                
                final_failed = len(retry_actions)
                if final_failed > 0:
                    self.logger.error(f"{final_failed} items failed permanently")
                    for a in retry_actions[:5]:
                        self.logger.error(f"  Failed doc _id: {a.get('_id')}")
                
                return (success + recovered, final_failed)
                
            except Exception as e:
                status = getattr(e, 'status_code', None)
                if status == 429 and attempt < self.MAX_429_RETRIES:
                    wait = min(self.INITIAL_BACKOFF_SEC * (2 ** attempt), self.MAX_BACKOFF_SEC)
                    self.logger.warning(
                        f"429 Too Many Requests, waiting {wait}s "
                        f"(retry {attempt+1}/{self.MAX_429_RETRIES})"
                    )
                    time.sleep(wait)
                else:
                    self.logger.error(f"Bulk indexing failed: {e}")
                    return (0, len(actions))
        
        return (0, len(actions))
    
    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 2000,
        show_progress: bool = False,
        refresh: bool = False,
        bulk_delay_sec: float = 0.0
    ) -> bool:
        """
        Index documents into Elasticsearch with robust retry.
        
        - 429 → 지수 백오프 재시도 (최대 7회, 5~300초 대기)
        - Bulk 응답 내 개별 실패 → 자동 재시도 (최대 3회)
        - 모든 문서가 성공해야 True 반환 (Checkpoint 기반 재개의 안전성 보장)
        - Idempotent _id → 재전송 시 덮어쓰기 (중복 없음)
        
        Args:
            documents: List of documents to index
            batch_size: Batch size for bulk indexing
            show_progress: Whether to show progress logs
            refresh: If True, refresh index after indexing
            bulk_delay_sec: Bulk 요청 간 대기 시간(초, 429 완화용)
            
        Returns:
            True if ALL documents were indexed successfully
        """
        if not documents:
            self.logger.warning("No documents to index")
            return True
        
        total_indexed = 0
        total_failed = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Bulk 요청 간 대기 (429 완화)
            if bulk_delay_sec > 0 and i > 0:
                time.sleep(bulk_delay_sec)
            
            actions = self._prepare_actions(batch)
            batch_indexed, batch_failed = self._send_bulk_with_retry(actions)
            total_indexed += batch_indexed
            total_failed += batch_failed
            
            if show_progress:
                progress = (i + len(batch)) / len(documents) * 100
                self.logger.info(f"Progress: {progress:.1f}% ({total_indexed}/{len(documents)})")
        
        # Refresh (마지막에만)
        if refresh:
            try:
                self.es.indices.refresh(index=self.index_name)
            except Exception as e:
                self.logger.warning(f"Refresh failed (non-critical): {e}")
        
        self.logger.info(
            f"Batch result: {total_indexed} indexed, {total_failed} failed "
            f"/ {len(documents)} total"
        )
        
        # 모든 문서가 성공한 경우에만 True (Checkpoint 진행 조건)
        return total_failed == 0
    
    def search_by_embedding(
        self,
        query_embedding: List[float],
        size: int = 10,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            size: Number of results to return
            min_score: Minimum similarity score
            
        Returns:
            List of matching documents
        """
        if not self.include_embeddings:
            self.logger.warning("Embeddings disabled, cannot perform vector search")
            return []
        
        try:
            # Determine embedding field name
            if "synonym" in self.index_name.lower():
                field = "concept_synonym_embedding"
            else:
                field = "concept_embedding"
            
            query = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": f"cosineSimilarity(params.query_vector, '{field}') + 1.0",
                            "params": {"query_vector": query_embedding}
                        }
                    }
                },
                "min_score": min_score + 1.0,
                "size": size
            }
            
            response = self.es.search(index=self.index_name, body=query)
            
            results = []
            for hit in response["hits"]["hits"]:
                result = hit["_source"].copy()
                result["_score"] = hit["_score"] - 1.0  # Restore original score
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Vector search error: {e}")
            return []
    
    def search_by_text(
        self,
        query: str,
        size: int = 10,
        field: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search using text matching.
        
        Args:
            query: Search query text
            size: Number of results to return
            field: Field to search (auto-detected if None)
            
        Returns:
            List of matching documents
        """
        try:
            # Determine search fields
            if field:
                fields = [field]
            elif "synonym" in self.index_name.lower():
                fields = ["concept_synonym_name^2", "concept_synonym_name.trigram"]
            else:
                fields = ["concept_name^2", "concept_name.trigram"]
            
            search_query = {
                    "query": {
                        "multi_match": {
                            "query": query,
                        "fields": fields,
                            "type": "best_fields",
                            "fuzziness": "AUTO"
                        }
                    },
                    "size": size
                }
            
            response = self.es.search(index=self.index_name, body=search_query)
            
            return [hit["_source"] for hit in response["hits"]["hits"]]
            
        except Exception as e:
            self.logger.error(f"Text search error: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics with 429 retry."""
        for attempt in range(self.MAX_429_RETRIES + 1):
            try:
                stats = self.es.indices.stats(index=self.index_name)
                return {
                    "document_count": stats["indices"][self.index_name]["total"]["docs"]["count"],
                    "store_size_bytes": stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"],
                    "index_name": self.index_name
                }
            except Exception as e:
                status = getattr(e, 'status_code', None)
                if status == 429 and attempt < self.MAX_429_RETRIES:
                    wait = min(self.INITIAL_BACKOFF_SEC * (2 ** attempt), self.MAX_BACKOFF_SEC)
                    self.logger.warning(f"429 on get_stats, waiting {wait}s")
                    time.sleep(wait)
                else:
                    self.logger.error(f"Failed to get stats: {e}")
                    return {}
        return {}
    
    def delete_index(self) -> bool:
        """Delete the index."""
        try:
            if self.es.indices.exists(index=self.index_name):
                self.es.indices.delete(index=self.index_name)
                self.logger.info(f"Index deleted: {self.index_name}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to delete index: {e}")
            return False


# Backward compatibility alias
ConceptElasticsearchIndexer = ElasticsearchIndexer
