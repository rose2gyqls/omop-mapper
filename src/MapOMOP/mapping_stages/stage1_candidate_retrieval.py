"""
Stage 1: Candidate Retrieval

Retrieves candidate concepts using multiple search strategies:
- Lexical: Text-based search (exact match, phrase match, fuzzy)
- Semantic: Vector-based search (SapBERT embeddings)
- Combined: Hybrid search (text + vector + length similarity)

Note: Synonyms are searched but converted to original concepts for return.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils import sigmoid_normalize

logger = logging.getLogger(__name__)


class Stage1CandidateRetrieval:
    """Stage 1: Multi-strategy candidate retrieval."""
    
    # Threshold settings
    LEXICAL_THRESHOLD = 3.0
    SEMANTIC_THRESHOLD = 0.8
    COMBINED_THRESHOLD = 3.0
    
    def __init__(self, es_client, has_sapbert: bool = True):
        """
        Initialize Stage 1.
        
        Args:
            es_client: Elasticsearch client
            has_sapbert: Whether SapBERT is available
        """
        self.es_client = es_client
        self.has_sapbert = has_sapbert
    
    def retrieve_candidates(
        self,
        entity_name: str,
        domain_id: str,
        entity_embedding: Optional[np.ndarray] = None,
        es_index: str = "concept"
    ) -> List[Dict[str, Any]]:
        """
        Retrieve candidates using multiple search strategies.
        
        Uses Lexical + Semantic + Combined search strategies.
        Synonyms are converted to original concepts while preserving search scores.
        
        Args:
            entity_name: Entity name to search
            domain_id: Domain filter
            entity_embedding: Entity embedding vector (optional)
            es_index: Elasticsearch index name
            
        Returns:
            List of candidate hits with _search_type field (all converted to Original)
        """
        logger.info("=" * 60)
        logger.info("Stage 1: Candidate Retrieval (Lexical + Semantic + Combined)")
        logger.info(f"  Entity: {entity_name}")
        logger.info(f"  Domain: {domain_id}")
        logger.info("=" * 60)
        
        all_candidates = []
        
        # 1. Lexical Search
        logger.info(f"\n[1/3] Lexical Search (threshold: {self.LEXICAL_THRESHOLD})")
        lexical_results = self._lexical_search(entity_name, domain_id, es_index, 3)
        lexical_filtered = [h for h in lexical_results if h['_score'] >= self.LEXICAL_THRESHOLD]
        
        logger.info(f"  Results: {len(lexical_results)} -> {len(lexical_filtered)} (passed threshold)")
        
        # Convert synonyms to original concepts
        lexical_converted = self._convert_synonyms_to_original(lexical_filtered, es_index)
        
        for hit in lexical_converted:
            hit['_search_type'] = 'lexical'
            all_candidates.append(hit)
        
        # 2. Semantic Search
        semantic_filtered = []
        semantic_converted = []
        if entity_embedding is not None:
            logger.info(f"\n[2/3] Semantic Search (threshold: {self.SEMANTIC_THRESHOLD})")
            semantic_results = self._vector_search(entity_embedding, domain_id, es_index, 3)
            semantic_filtered = [h for h in semantic_results if h['_score'] >= self.SEMANTIC_THRESHOLD]
            
            logger.info(f"  Results: {len(semantic_results)} -> {len(semantic_filtered)} (passed threshold)")
            
            # Convert synonyms to original concepts
            semantic_converted = self._convert_synonyms_to_original(semantic_filtered, es_index)
            
            for hit in semantic_converted:
                hit['_search_type'] = 'semantic'
                all_candidates.append(hit)
        else:
            logger.info("\n[2/3] Semantic Search - Skipped (no embedding)")
        
        # 3. Combined/Hybrid Search
        combined_filtered = []
        combined_converted = []
        logger.info(f"\n[3/3] Combined Search (threshold: {self.COMBINED_THRESHOLD})")
        
        if entity_embedding is not None:
            combined_results = self._hybrid_search(entity_name, entity_embedding, domain_id, es_index, 3)
        else:
            combined_results = lexical_filtered[:3]
        
        combined_filtered = [h for h in combined_results if h['_score'] >= self.COMBINED_THRESHOLD]
        
        logger.info(f"  Results: {len(combined_results)} -> {len(combined_filtered)} (passed threshold)")
        
        # Convert synonyms to original concepts
        combined_converted = self._convert_synonyms_to_original(combined_filtered, es_index)
        
        for hit in combined_converted:
            hit['_search_type'] = 'combined'
            all_candidates.append(hit)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info(f"Stage 1 Complete: {len(all_candidates)} total candidates")
        logger.info(f"  - Lexical: {len(lexical_converted)}")
        logger.info(f"  - Semantic: {len(semantic_converted)}")
        logger.info(f"  - Combined: {len(combined_converted)}")
        logger.info("=" * 60)
        
        return all_candidates
    
    def _get_domain_filter(self, domain_id: str) -> Dict:
        """Build domain filter query."""
        return {"term": {"domain_id": domain_id}}
    
    def _convert_synonyms_to_original(
        self,
        hits: List[Dict[str, Any]],
        es_index: str
    ) -> List[Dict[str, Any]]:
        """
        Convert synonym hits to original concept hits.
        
        For hits where name_type='Synonym', fetch the original concept
        (name_type='Original') and replace the source while preserving the score.
        
        Args:
            hits: List of ES hits
            es_index: Elasticsearch index name
            
        Returns:
            List of hits with synonyms converted to original concepts
        """
        if not hits:
            return []
        
        # Separate original and synonym hits
        original_hits = []
        synonym_hits = []
        
        for hit in hits:
            name_type = hit['_source'].get('name_type', 'Original')
            if name_type == 'Synonym':
                synonym_hits.append(hit)
            else:
                original_hits.append(hit)
        
        if not synonym_hits:
            return hits  # No synonyms to convert
        
        # Collect concept_ids from synonym hits
        synonym_concept_ids = [str(h['_source'].get('concept_id', '')) for h in synonym_hits]
        synonym_concept_ids = list(set(synonym_concept_ids))  # Deduplicate
        
        logger.debug(f"Converting {len(synonym_hits)} synonym hits to original concepts")
        
        # Fetch original concepts for these concept_ids
        original_concepts = self._fetch_original_concepts(synonym_concept_ids, es_index)
        
        # Create a map of concept_id -> original concept data
        original_map = {}
        for concept in original_concepts:
            cid = str(concept.get('concept_id', ''))
            if cid:
                original_map[cid] = concept
        
        # Convert synonym hits to original concept hits
        converted_hits = []
        for hit in synonym_hits:
            concept_id = str(hit['_source'].get('concept_id', ''))
            original_concept = original_map.get(concept_id)
            
            if original_concept:
                # Preserve search metadata but replace source with original concept
                matched_synonym_name = hit['_source'].get('concept_name', '')
                
                converted_hit = {
                    '_index': hit['_index'],
                    '_id': hit['_id'],
                    '_score': hit['_score'],  # Preserve synonym search score
                    '_score_normalized': hit.get('_score_normalized'),
                    '_source': original_concept,  # Replace with original concept
                    '_matched_synonym': matched_synonym_name  # Store matched synonym for reference
                }
                converted_hits.append(converted_hit)
                
                logger.debug(f"  Synonym '{matched_synonym_name}' -> Original '{original_concept.get('concept_name', '')}'")
            else:
                # If original not found, keep the synonym hit as-is
                logger.warning(f"Original concept not found for concept_id: {concept_id}")
                converted_hits.append(hit)
        
        # Combine original hits and converted synonym hits
        all_hits = original_hits + converted_hits
        
        return all_hits
    
    def _fetch_original_concepts(
        self,
        concept_ids: List[str],
        es_index: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch original concepts (name_type='Original') by concept_ids.
        
        Args:
            concept_ids: List of concept IDs to fetch
            es_index: Elasticsearch index name
            
        Returns:
            List of original concept documents
        """
        if not concept_ids:
            return []
        
        try:
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"terms": {"concept_id": concept_ids}},
                            {"term": {"name_type": "Original"}}
                        ]
                    }
                },
                "size": len(concept_ids)
            }
            
            response = self.es_client.es_client.search(index=es_index, body=query)
            
            return [hit['_source'] for hit in response['hits']['hits']]
            
        except Exception as e:
            logger.error(f"Failed to fetch original concepts: {e}")
            return []
    
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
                                {"match": {"concept_name": {
                                    "query": entity_name,
                                    "minimum_should_match": "1<60%",
                                    "boost": 2.0
                                }}}
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
            hits = response['hits']['hits'] if response['hits']['total']['value'] > 0 else []
            # Add sigmoid normalized score
            for hit in hits:
                hit['_score_normalized'] = sigmoid_normalize(hit['_score'], center=3.0)
            return hits
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
                                        {"match": {"concept_name": {
                                            "query": entity_name,
                                            "minimum_should_match": "1<60%",
                                            "boost": 2.5
                                        }}}
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
            hits = response['hits']['hits'] if response['hits']['total']['value'] > 0 else []
            # Add sigmoid normalized score
            for hit in hits:
                hit['_score_normalized'] = sigmoid_normalize(hit['_score'], center=3.0)
            return hits
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return self._lexical_search(entity_name, domain_id, es_index, top_k)
