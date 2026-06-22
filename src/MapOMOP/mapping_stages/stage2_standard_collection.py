"""
Stage 2: Standard Concept Collection

Converts concepts using relationship transformations and "Maps to" relationships.
Process (2 rounds of [relation transform → Maps to]):
1. Round 1: Stage 1 candidates → transform via TRANSFORM_RELATIONSHIP_IDS
   → for both transformed and untransformed candidates, apply Maps to if non-std
   → only std results are passed to round 2
2. Round 2: std results → transform via TRANSFORM_RELATIONSHIP_IDS
   → for both transformed and untransformed candidates, apply Maps to if non-std
3. Merge round 1 and round 2 results and deduplicate
"""

import logging
from typing import Any, Dict, List, Optional, Set

from ..utils import deduplicate_by_concept

logger = logging.getLogger(__name__)


class Stage2StandardCollection:
    """Stage 2: Convert to standard concepts via relationship transformations."""
    
    # Relationship IDs for transformation
    TRANSFORM_RELATIONSHIP_IDS = [
        'Concept alt_to to',
        'Concept poss_eq to',
        'Concept same_as to',
        'Marketed form of',
        'Tradename of',
        'Box of',
        'Has quantified form',
        'Is a'
    ]
    
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
        1. Round 1: relation transform → Maps to for both transformed/untransformed → only std to round 2
        2. Round 2: repeat the same process on std results
        3. Deduplicate all results
        
        Args:
            stage1_candidates: Candidates from Stage 1
            domain_id: Domain ID for logging
            
        Returns:
            Deduplicated standard candidates
        """
        logger.info("Stage 2: Standard Concept Collection")
        
        initial_candidates = self._prepare_initial_candidates(stage1_candidates)
        
        # Round 1: relation transform → Maps to
        first_round_all = self._apply_transform_and_maps_to(initial_candidates)
        first_round_dedup = deduplicate_by_concept(
            first_round_all,
            get_concept=lambda c: c.get('concept', {}),
            get_score=lambda c: c.get('elasticsearch_score', 0.0)
        )
        
        # Round 2: feed only std results as input, same process
        first_round_std = self._filter_std_only(first_round_dedup)
        second_round_all = (
            self._apply_transform_and_maps_to(first_round_std)
            if first_round_std else []
        )
        second_round_dedup = deduplicate_by_concept(
            second_round_all,
            get_concept=lambda c: c.get('concept', {}),
            get_score=lambda c: c.get('elasticsearch_score', 0.0)
        ) if second_round_all else []
        
        # Merge round 1 + round 2, deduplicate, then pass to stage 3
        all_results = first_round_dedup + second_round_dedup
        deduplicated = deduplicate_by_concept(
            all_results,
            get_concept=lambda c: c.get('concept', {}),
            get_score=lambda c: c.get('elasticsearch_score', 0.0)
        )
        
        self._log_candidates_detail("Round 1 results", first_round_dedup)
        self._log_candidates_detail("Round 2 results", second_round_dedup)
        self._log_candidates_detail("Deduplicated results", deduplicated)
        
        return deduplicated
    
    def _prepare_initial_candidates(
        self,
        stage1_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prepare initial candidate structure from Stage 1 results."""
        candidates = []
        
        for candidate in stage1_candidates:
            source = candidate['_source']
            search_type = candidate.get('_search_type', 'unknown')
            
            candidates.append({
                'concept': source,
                'is_original_standard': source.get('standard_concept') in ['S', 'C'],
                'original_candidate': candidate,
                'original_non_standard': None,
                'relation_type': 'original',
                'elasticsearch_score': candidate.get('_score_normalized') or candidate['_score'],
                'search_type': search_type
            })
        
        return candidates
    
    def _log_candidates_detail(self, title: str, candidates: List[Dict[str, Any]]):
        """Log transform results: concept_name (concept_id) [relation_type] ← from source (shows transform relation)."""
        logger.info(f"  [{title}]")
        if not candidates:
            logger.info("    (none)")
            return
        for c in candidates:
            concept = c.get('concept', {})
            name = concept.get('concept_name', 'N/A')
            cid = concept.get('concept_id', 'N/A')
            rel = c.get('relation_type', 'original')
            ons = c.get('original_non_standard')
            if ons:
                from_name = ons.get('concept_name', 'N/A')
                logger.info(f"    {name} ({cid}) [{rel}] ← {from_name}")
            else:
                logger.info(f"    {name} ({cid}) [{rel}]")
    
    def _apply_transform_and_maps_to(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply one round of [relation transform → Maps to].
        
        - Transform via the relationships specified in TRANSFORM_RELATIONSHIP_IDS
        - For both transformed candidates and untransformed originals:
          if std, add as-is; if non-std, find std via Maps to and add
        """
        result = []
        
        for candidate in candidates:
            concept = candidate['concept']
            concept_id = str(concept.get('concept_id', ''))
            
            # 1) Original: std as-is, non-std via Maps to
            self._add_candidate_or_maps_to(
                candidate, concept, candidate.get('original_non_standard'), 'original', result
            )
            
            # 2) Relationship transform
            related_concepts = self._get_related_concepts_with_relation(concept_id)
            for related, relation_id in related_concepts:
                related_id = str(related.get('concept_id', ''))
                self._add_candidate_or_maps_to(
                    {'concept': related, 'original_candidate': candidate['original_candidate'],
                     'search_type': candidate['search_type']},
                    related,
                    concept,
                    relation_id,
                    result
                )
        
        return result
    
    def _add_candidate_or_maps_to(
        self,
        candidate_data: Dict[str, Any],
        concept: Dict[str, Any],
        original_non_std: Optional[Dict[str, Any]],
        relation_type: str,
        result: List[Dict[str, Any]]
    ) -> None:
        """
        If std, add as-is; if non-std, find std via Maps to and add.
        """
        is_std = concept.get('standard_concept') in ['S', 'C']
        
        if is_std:
            result.append({
                'concept': concept,
                'is_original_standard': candidate_data.get('is_original_standard', False),
                'original_candidate': candidate_data.get('original_candidate'),
                'original_non_standard': original_non_std,
                'relation_type': relation_type,
                'elasticsearch_score': candidate_data.get('elasticsearch_score', 0.0),
                'search_type': candidate_data.get('search_type', 'unknown')
            })
        else:
            concept_id = str(concept.get('concept_id', ''))
            std_concepts = self._get_standard_via_maps_to(concept_id)
            if std_concepts:
                for std in std_concepts:
                    result.append({
                        'concept': std,
                        'is_original_standard': False,
                        'original_candidate': candidate_data.get('original_candidate'),
                        'original_non_standard': concept,
                        'relation_type': 'Maps to',
                        'elasticsearch_score': 0.0,
                        'search_type': candidate_data.get('search_type', 'unknown')
                    })
    
    def _filter_std_only(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return only std results (standard_concept in S/C)."""
        return [
            c for c in candidates
            if c.get('concept', {}).get('standard_concept') in ['S', 'C']
        ]
    
    def _get_related_concepts_with_relation(
        self, concept_id: str
    ) -> List[tuple]:
        """
        Get related concepts via TRANSFORM_RELATIONSHIP_IDS with relationship_id.
        
        Args:
            concept_id: Source concept ID (concept_id_1)
            
        Returns:
            List of (concept_dict, relationship_id) tuples
        """
        try:
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"concept_id_1": concept_id}},
                            {"terms": {"relationship_id": self.TRANSFORM_RELATIONSHIP_IDS}}
                        ]
                    }
                },
                "size": 50
            }
            
            response = self.es_client.es_client.search(
                index="concept-relationship",
                body=query
            )
            
            # Collect (concept_id_2, relationship_id) - first occurrence per concept_id_2
            id_to_relation = {}
            for hit in response['hits']['hits']:
                src = hit['_source']
                concept_id_2 = src.get('concept_id_2')
                rel_id = src.get('relationship_id', 'unknown')
                if concept_id_2 and str(concept_id_2) not in id_to_relation:
                    id_to_relation[str(concept_id_2)] = rel_id
            
            if not id_to_relation:
                return []
            
            # Fetch concept details
            concept_ids = list(id_to_relation.keys())
            concepts = self._search_concepts_by_ids(concept_ids, require_standard=False)
            
            return [(c, id_to_relation.get(str(c.get('concept_id', '')), 'unknown'))
                    for c in concepts]
            
        except Exception as e:
            logger.warning(f"Relationship lookup failed for {concept_id}: {e}")
            return []
    
    def _get_standard_via_maps_to(self, concept_id: str) -> List[Dict[str, Any]]:
        """
        Get standard concepts via "Maps to" relationship.
        
        Args:
            concept_id: Non-standard concept ID
            
        Returns:
            List of standard concepts
        """
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
            
            standard_ids = []
            for hit in response['hits']['hits']:
                concept_id_2 = hit['_source'].get('concept_id_2')
                if concept_id_2:
                    standard_ids.append(str(concept_id_2))
            
            if not standard_ids:
                return []
            
            # Fetch standard concepts only
            return self._search_concepts_by_ids(standard_ids, require_standard=True)
            
        except Exception as e:
            logger.warning(f"Maps to lookup failed for {concept_id}: {e}")
            return []
    
    def _search_concepts_by_ids(
        self,
        concept_ids: List[str],
        require_standard: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search concepts by IDs.
        
        Args:
            concept_ids: List of concept IDs
            require_standard: If True, only return standard concepts (S/C)
            
        Returns:
            List of concept documents
        """
        if not concept_ids:
            return []
        
        try:
            must_conditions = [
                {"terms": {"concept_id": concept_ids}},
                {"term": {"name_type": "Original"}}
            ]
            
            if require_standard:
                must_conditions.append({"terms": {"standard_concept": ["S", "C"]}})
            
            query = {
                "query": {
                    "bool": {
                        "must": must_conditions
                    }
                },
                "size": len(concept_ids)
            }
            
            response = self.es_client.es_client.search(
                index=self.es_client.concept_index,
                body=query
            )
            
            return [hit['_source'] for hit in response['hits']['hits']]
            
        except Exception as e:
            logger.warning(f"Concept search failed: {e}")
            return []
    
