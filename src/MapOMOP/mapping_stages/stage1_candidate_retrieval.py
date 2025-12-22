"""
Stage 1: Candidate Retrieval

Retrieves candidate concepts using multiple search strategies:
- Lexical: Text-based search (exact match, phrase match, fuzzy)
- Semantic: Vector-based search (SapBERT embeddings)
- Combined: Hybrid search (text + vector + length similarity)
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Stage1CandidateRetrieval:
    """Stage 1: Multi-strategy candidate retrieval."""
    
    # Threshold settings
    LEXICAL_THRESHOLD = 5.0
    SEMANTIC_THRESHOLD = 0.8
    COMBINED_THRESHOLD = 5.0
    
    def __init__(
        self,
        es_client,
        has_sapbert: bool = True,
        use_lexical: bool = True
    ):
        """
        Initialize Stage 1.
        
        Args:
            es_client: Elasticsearch client
            has_sapbert: Whether SapBERT is available
            use_lexical: Whether to use lexical search
        """
        self.es_client = es_client
        self.has_sapbert = has_sapbert
        self.use_lexical = use_lexical
    
    def retrieve_candidates(
        self,
        entity_name: str,
        domain_id: str,
        entity_embedding: Optional[np.ndarray] = None,
        es_index: str = "concept"
    ) -> List[Dict[str, Any]]:
        """
        Retrieve candidates using multiple search strategies.
        
        Args:
            entity_name: Entity name to search
            domain_id: Domain filter
            entity_embedding: Entity embedding vector (optional)
            es_index: Elasticsearch index name
            
        Returns:
            List of candidate hits with _search_type field
        """
        mode = "Lexical + Semantic + Combined" if self.use_lexical else "Semantic + Combined"
        
        logger.info("=" * 60)
        logger.info(f"Stage 1: Candidate Retrieval ({mode})")
        logger.info(f"  Entity: {entity_name}")
        logger.info(f"  Domain: {domain_id}")
        logger.info("=" * 60)
        
        all_candidates = []
        
        # 1. Lexical Search
        lexical_filtered = []
        if self.use_lexical:
            logger.info(f"\n[1/3] Lexical Search (threshold: {self.LEXICAL_THRESHOLD})")
            lexical_results = self._lexical_search(entity_name, domain_id, es_index, 3)
            lexical_filtered = [h for h in lexical_results if h['_score'] >= self.LEXICAL_THRESHOLD]
            
            logger.info(f"  Results: {len(lexical_results)} -> {len(lexical_filtered)} (passed threshold)")
            
            for hit in lexical_filtered:
                hit['_search_type'] = 'lexical'
                all_candidates.append(hit)
        else:
            logger.info("\n[1/3] Lexical Search - Skipped (use_lexical=False)")
        
        # 2. Semantic Search
        semantic_filtered = []
        if entity_embedding is not None:
            logger.info(f"\n[2/3] Semantic Search (threshold: {self.SEMANTIC_THRESHOLD})")
            semantic_results = self._vector_search(entity_embedding, domain_id, es_index, 3)
            semantic_filtered = [h for h in semantic_results if h['_score'] >= self.SEMANTIC_THRESHOLD]
            
            logger.info(f"  Results: {len(semantic_results)} -> {len(semantic_filtered)} (passed threshold)")
            
            for hit in semantic_filtered:
                hit['_search_type'] = 'semantic'
                all_candidates.append(hit)
        else:
            logger.info("\n[2/3] Semantic Search - Skipped (no embedding)")
        
        # 3. Combined/Hybrid Search
        combined_filtered = []
        logger.info(f"\n[3/3] Combined Search (threshold: {self.COMBINED_THRESHOLD})")
        
        if entity_embedding is not None:
            combined_results = self._hybrid_search(entity_name, entity_embedding, domain_id, es_index, 3)
        else:
            combined_results = lexical_filtered[:3] if self.use_lexical else []
        
        combined_filtered = [h for h in combined_results if h['_score'] >= self.COMBINED_THRESHOLD]
        
        logger.info(f"  Results: {len(combined_results)} -> {len(combined_filtered)} (passed threshold)")
        
        for hit in combined_filtered:
            hit['_search_type'] = 'combined'
            all_candidates.append(hit)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info(f"Stage 1 Complete: {len(all_candidates)} total candidates")
        if self.use_lexical:
            logger.info(f"  - Lexical: {len(lexical_filtered)}")
        logger.info(f"  - Semantic: {len(semantic_filtered)}")
        logger.info(f"  - Combined: {len(combined_filtered)}")
        logger.info("=" * 60)
        
        return all_candidates
    
    def _get_domain_filter(self, domain_id: str) -> Dict:
        """Build domain filter query."""
        # Handle Measurement domain (includes "Meas Value")
        if domain_id == "Measurement":
            return {"terms": {"domain_id": ["Measurement", "Meas Value"]}}
        return {"term": {"domain_id": domain_id}}
    
    def _lexical_search(
        self,
        entity_name: str,
        domain_id: str,
        es_index: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform text-based search."""
        query = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [{
                        "bool": {
                            "should": [
                                {"term": {"concept_name.keyword": {"value": entity_name, "boost": 3.0}}},
                                {"match_phrase": {"concept_name": {"query": entity_name, "boost": 2.5}}},
                                {"match": {"concept_name": {"query": entity_name, "boost": 2.0}}}
                            ],
                            "minimum_should_match": 1
                        }
                    }],
                    "filter": [self._get_domain_filter(domain_id)]
                }
            }
        }
        
        try:
            response = self.es_client.es_client.search(index=es_index, body=query)
            return response['hits']['hits'] if response['hits']['total']['value'] > 0 else []
        except Exception as e:
            logger.error(f"Lexical search failed: {e}")
            return []
    
    def _vector_search(
        self,
        entity_embedding: np.ndarray,
        domain_id: str,
        es_index: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform vector-based search."""
        query = {
            "knn": {
                "field": "concept_embedding",
                "query_vector": entity_embedding.tolist(),
                "k": top_k,
                "num_candidates": top_k * 3,
                "filter": self._get_domain_filter(domain_id)
            },
            "size": top_k,
            "_source": True
        }
        
        try:
            response = self.es_client.es_client.search(index=es_index, body=query)
            return response['hits']['hits'] if response['hits']['total']['value'] > 0 else []
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _hybrid_search(
        self,
        entity_name: str,
        entity_embedding: np.ndarray,
        domain_id: str,
        es_index: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search (text + vector + length similarity)."""
        entity_length = len(entity_name.strip())
        scale_len = max(8.0, entity_length * 0.8)
        domain_filter = self._get_domain_filter(domain_id)
        
        query = {
            "size": top_k,
            "knn": {
                "field": "concept_embedding",
                "query_vector": entity_embedding.tolist(),
                "k": top_k * 2,
                "num_candidates": top_k * 5,
                "boost": 0.6,
                "filter": domain_filter
            },
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "must": [{
                                "bool": {
                                    "should": [
                                        {"term": {"concept_name.keyword": {"value": entity_name, "boost": 3.0}}},
                                        {"match": {"concept_name": {"query": entity_name, "boost": 2.5}}}
                                    ],
                                    "minimum_should_match": 1
                                }
                            }],
                            "filter": [domain_filter]
                        }
                    },
                    "functions": [{
                        "script_score": {
                            "script": {
                                "params": {"origin_len": float(entity_length), "scale_len": float(scale_len)},
                                "source": """
                                    double origin = params.origin_len;
                                    double scale = params.scale_len;
                                    double len = 0.0;
                                    if (!doc['concept_name.keyword'].isEmpty()) {
                                        len = doc['concept_name.keyword'].value.length();
                                    }
                                    double x = (len - origin) / scale;
                                    return 1.0 + Math.exp(-0.5 * x * x);
                                """
                            }
                        }
                    }],
                    "score_mode": "multiply",
                    "boost_mode": "multiply",
                    "boost": 0.4
                }
            }
        }
        
        try:
            response = self.es_client.es_client.search(index=es_index, body=query)
            return response['hits']['hits'] if response['hits']['total']['value'] > 0 else []
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return self._lexical_search(entity_name, domain_id, es_index, top_k)
