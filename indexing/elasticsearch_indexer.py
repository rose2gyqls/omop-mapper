"""
Elasticsearch Indexer Module

Provides functionality to index OMOP CDM data into Elasticsearch.
Supports CONCEPT, CONCEPT_RELATIONSHIP, and CONCEPT_SYNONYM tables.
"""

import logging
import hashlib
from typing import Dict, List, Optional, Any

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


class ElasticsearchIndexer:
    """Elasticsearch indexer for OMOP CDM data."""
    
    # Index mapping configurations
    MAPPINGS = {
        'concept': {
                    "mappings": {
                        "properties": {
                            "concept_id": {"type": "keyword"},
                            "concept_name": {
                                "type": "text",
                                "fields": {
                            "keyword": {"type": "keyword"},
                            "trigram": {"type": "text", "analyzer": "trigram_analyzer"}
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
                    "concept_synonym_name": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "trigram": {"type": "text", "analyzer": "trigram_analyzer"}
                        }
                    },
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
        
        # Test connection
        if not self.es.ping():
            raise ConnectionError(f"Failed to connect to Elasticsearch at {host}:{port}")
        
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
    
    def _get_index_mapping(self) -> Dict:
        """Get appropriate mapping for the index type."""
        # Determine index type from name
        if "relationship" in self.index_name.lower():
            mapping = self.MAPPINGS['concept-relationship'].copy()
        elif "synonym" in self.index_name.lower():
            mapping = self.MAPPINGS['concept-synonym'].copy()
        else:
            mapping = self.MAPPINGS['concept'].copy()
        
        # Add settings
        mapping["settings"] = self.INDEX_SETTINGS.copy()
        
        # Add embedding field if enabled
        if self.include_embeddings:
            embedding_field = {
                    "type": "dense_vector",
                    "dims": 768,
                    "index": True,
                    "similarity": "cosine"
            }
        
        if "synonym" in self.index_name.lower():
            mapping["mappings"]["properties"]["concept_synonym_embedding"] = embedding_field
        elif "relationship" not in self.index_name.lower():
            mapping["mappings"]["properties"]["concept_embedding"] = embedding_field
        
        return mapping
    
    def create_index(self, delete_if_exists: bool = False) -> bool:
        """
        Create Elasticsearch index.
        
        Args:
            delete_if_exists: Delete existing index if it exists
            
        Returns:
            True if index was created successfully
        """
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
            import time
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
            self.logger.error(f"Failed to create index: {e}")
            return False
    
    def _generate_doc_id(self, doc: Dict) -> str:
        """Generate document ID based on index type."""
        if "relationship" in self.index_name.lower():
            unique_str = f"{doc.get('concept_id_1', '')}_{doc.get('concept_id_2', '')}_{doc.get('relationship_id', '')}"
            return hashlib.md5(unique_str.encode()).hexdigest()
        elif "synonym" in self.index_name.lower():
            unique_str = f"{doc.get('concept_id', '')}_{doc.get('concept_synonym_name', '')}"
            return hashlib.md5(unique_str.encode()).hexdigest()
        else:
            return str(doc.get("concept_id", ""))
    
    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 500,
        show_progress: bool = False
    ) -> bool:
        """
        Index documents into Elasticsearch.
        
        Args:
            documents: List of documents to index
            batch_size: Batch size for bulk indexing
            show_progress: Whether to show progress logs
            
        Returns:
            True if indexing was successful (>80% success rate)
        """
        if not documents:
            self.logger.warning("No documents to index")
            return True
        
        try:
            total_indexed = 0
            failed_count = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Prepare bulk actions
                actions = []
                for doc in batch:
                    # Remove embedding field if disabled
                    if not self.include_embeddings:
                        doc = doc.copy()
                        doc.pop("concept_embedding", None)
                        doc.pop("concept_synonym_embedding", None)
                    
                    actions.append({
                        "_index": self.index_name,
                        "_id": self._generate_doc_id(doc),
                        "_source": doc
                    })
                
                # Execute bulk indexing
                try:
                    success, failed = bulk(
                        client=self.es,
                        actions=actions,
                        index=self.index_name,
                        chunk_size=min(batch_size, 50),
                        request_timeout=300,
                        raise_on_error=False,
                        raise_on_exception=False,
                        max_retries=3
                    )
                    total_indexed += success
                    
                    if failed:
                        failed_count += len(failed)
                        if show_progress:
                            self.logger.warning(f"Batch failed: {len(failed)} documents")
                            
                except Exception as e:
                    self.logger.error(f"Bulk indexing error: {e}")
                    failed_count += len(batch)
                
                if show_progress:
                    progress = (i + batch_size) / len(documents) * 100
                    self.logger.info(f"Progress: {progress:.1f}% ({total_indexed}/{len(documents)})")
            
            # Refresh index
            self.es.indices.refresh(index=self.index_name)
            
            self.logger.info(f"Indexing complete: {total_indexed} documents indexed, {failed_count} failed")
            
            # Consider success if >80% indexed
            success_rate = total_indexed / len(documents) if documents else 1.0
            return success_rate >= 0.8
            
        except Exception as e:
            self.logger.error(f"Indexing error: {e}")
            return False
    
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
        """Get index statistics."""
        try:
            stats = self.es.indices.stats(index=self.index_name)
            return {
                "document_count": stats["indices"][self.index_name]["total"]["docs"]["count"],
                "store_size_bytes": stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"],
                "index_name": self.index_name
            }
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
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
