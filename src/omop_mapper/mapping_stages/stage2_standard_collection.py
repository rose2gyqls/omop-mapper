"""
Stage 2: Standard Concept Collection

Converts non-standard concepts to standard concepts using "Maps to" relationships.
Deduplicates candidates by concept_id and concept_name.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class Stage2StandardCollection:
    """Stage 2: Convert to standard concepts and deduplicate."""
    
    def __init__(self, es_client):
        """
        Initialize Stage 2.
        
        Args:
            es_client: Elasticsearch client
        """
        self.es_client = es_client
    
    def collect_standard_candidates(
        self,
        stage1_candidates: List[Dict[str, Any]],
        domain_id: str
    ) -> List[Dict[str, Any]]:
        """
        Convert Stage 1 candidates to standard concepts.
        
        Process:
        1. Standard concepts (S/C): Add directly
        2. Non-standard: Find standard via "Maps to" relationship
        3. Deduplicate by concept_id + concept_name
        
        Args:
            stage1_candidates: Candidates from Stage 1
            domain_id: Domain ID for logging
            
        Returns:
            Deduplicated standard candidates
        """
        logger.info("=" * 60)
        logger.info("Stage 2: Standard Concept Collection")
        logger.info("=" * 60)
        
        all_standards = []
        standard_count = 0
        non_standard_count = 0
        
        for candidate in stage1_candidates:
            source = candidate['_source']
            search_type = candidate.get('_search_type', 'unknown')
            
            if source.get('standard_concept') in ['S', 'C']:
                # Standard concept - add directly
                standard_count += 1
                all_standards.append({
                    'concept': source,
                    'is_original_standard': True,
                    'original_candidate': candidate,
                    'elasticsearch_score': candidate['_score'],
                    'search_type': search_type
                })
                logger.info(f"  [Standard] {source.get('concept_name')} "
                           f"(ID: {source.get('concept_id')}) [{search_type}]")
            else:
                # Non-standard - find standard via "Maps to"
                non_standard_count += 1
                concept_id = str(source.get('concept_id', ''))
                logger.info(f"  [Non-std] {source.get('concept_name')} "
                           f"(ID: {concept_id}) [{search_type}]")
                
                standard_candidates = self._get_standard_for_non_standard(concept_id)
                
                for std in standard_candidates:
                    all_standards.append({
                        'concept': std,
                        'is_original_standard': False,
                        'original_non_standard': source,
                        'original_candidate': candidate,
                        'elasticsearch_score': 0.0,
                        'search_type': search_type
                    })
                    logger.info(f"    -> Mapped to: {std.get('concept_name')} "
                               f"(ID: {std.get('concept_id')})")
        
        logger.info(f"\nClassification:")
        logger.info(f"  - Standard: {standard_count}")
        logger.info(f"  - Non-standard: {non_standard_count}")
        logger.info(f"  - Total collected: {len(all_standards)}")
        
        # Deduplicate
        deduplicated = self._deduplicate(all_standards)
        
        logger.info(f"\nDeduplication: {len(all_standards)} -> {len(deduplicated)}")
        logger.info("=" * 60)
        
        return deduplicated
    
    def _get_standard_for_non_standard(self, concept_id: str) -> List[Dict[str, Any]]:
        """Find standard concepts for a non-standard concept."""
        try:
            # Get "Maps to" relationships
            standard_ids = self._get_maps_to_ids(concept_id)
            
            # Search for standard concepts
            return self._search_concepts_by_ids(standard_ids)
            
        except Exception as e:
            logger.error(f"Standard lookup failed: {e}")
            return []
    
    def _get_maps_to_ids(self, concept_id: str) -> List[str]:
        """Get concept IDs from "Maps to" relationships."""
        try:
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"concept_id_1": concept_id}},
                            {"match": {"relationship_id": "Maps to"}}
                        ]
                    }
                },
                "size": 10
            }
            
            response = self.es_client.es_client.search(
                index="concept-relationship",
                body=query
            )
            
            ids = []
            for hit in response['hits']['hits']:
                concept_id_2 = hit['_source'].get('concept_id_2')
                if concept_id_2:
                    ids.append(str(concept_id_2))
            
            if ids:
                logger.debug(f"Found {len(ids)} 'Maps to' relations for {concept_id}")
            
            return ids
            
        except Exception as e:
            logger.warning(f"Maps to lookup failed: {e}")
            return []
    
    def _search_concepts_by_ids(self, concept_ids: List[str]) -> List[Dict[str, Any]]:
        """Search concepts by IDs."""
        if not concept_ids:
            return []
        
        try:
            query = {
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
            
            response = self.es_client.es_client.search(
                index="concept",
                body=query
            )
            
            return [hit['_source'] for hit in response['hits']['hits']]
            
        except Exception as e:
            logger.warning(f"Concept search failed: {e}")
            return []
    
    def _deduplicate(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate candidates by concept_id + concept_name."""
        unique = {}
        
        for candidate in candidates:
            concept = candidate['concept']
            key = (concept.get('concept_id', ''), concept.get('concept_name', ''))
            
            if key not in unique:
                unique[key] = candidate
            else:
                # Keep higher score
                if candidate['elasticsearch_score'] > unique[key]['elasticsearch_score']:
                    unique[key] = candidate
        
        return list(unique.values())
