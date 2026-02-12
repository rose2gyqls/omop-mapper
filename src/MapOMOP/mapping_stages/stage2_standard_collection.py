"""
Stage 2: Standard Concept Collection

Converts concepts using relationship transformations and "Maps to" relationships.
Process:
1. 1차 변환: Stage 1 결과 → relationship 변환 → non-std면 Maps to
2. 2차 변환: 1차 결과 → relationship 변환
3. 3차 변환: 2차 결과 → non-std면 Maps to → 중복 삭제
"""

import logging
from typing import Any, Dict, List, Set

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
        'Has quantified form'
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
        1. 1차 변환: relationship 변환 후 non-std면 Maps to
        2. 2차 변환: 1차 결과에 relationship 변환
        3. 3차 변환: non-std면 Maps to, 중복 삭제
        
        Args:
            stage1_candidates: Candidates from Stage 1
            domain_id: Domain ID for logging
            
        Returns:
            Deduplicated standard candidates
        """
        logger.info("=" * 60)
        logger.info("Stage 2: Standard Concept Collection")
        logger.info("=" * 60)
        
        # Prepare initial candidates
        initial_candidates = self._prepare_initial_candidates(stage1_candidates)
        logger.info(f"Initial candidates: {len(initial_candidates)}")
        
        # 1차 변환: relationship 변환 → non-std면 Maps to
        logger.info("\n--- 1차 변환: Relationship + Maps to ---")
        first_transform = self._first_transform(initial_candidates)
        logger.info(f"After 1st transform: {len(first_transform)}")
        
        # 2차 변환: relationship 변환만
        logger.info("\n--- 2차 변환: Relationship only ---")
        second_transform = self._second_transform(first_transform)
        logger.info(f"After 2nd transform: {len(second_transform)}")
        
        # 3차 변환: non-std면 Maps to → 중복 삭제
        logger.info("\n--- 3차 변환: Maps to + Dedup ---")
        final_candidates = self._third_transform(second_transform)
        logger.info(f"After 3rd transform: {len(final_candidates)}")
        
        # Deduplicate
        deduplicated = self._deduplicate(final_candidates)
        
        logger.info(f"\nFinal deduplication: {len(final_candidates)} -> {len(deduplicated)}")
        logger.info("=" * 60)
        
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
            
            std_label = "S" if source.get('standard_concept') in ['S', 'C'] else "N"
            logger.debug(f"  [{std_label}] {source.get('concept_name')} "
                        f"(ID: {source.get('concept_id')}) [{search_type}]")
        
        return candidates
    
    def _first_transform(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        1차 변환: relationship 변환 후 non-std면 Maps to.
        
        - Stage 1 결과를 concept_id_1에서 찾고 relationship_id에 해당하면 변환
        - 변환된 것 중 non-std면 Maps to로 std 변환
        """
        result = []
        
        for candidate in candidates:
            concept = candidate['concept']
            concept_id = str(concept.get('concept_id', ''))
            
            # 원본 컨셉 추가
            result.append(candidate)
            
            # Relationship 변환 시도
            related_concepts = self._get_related_concepts_with_relation(concept_id)
            
            for related, relation_id in related_concepts:
                related_id = str(related.get('concept_id', ''))
                is_std = related.get('standard_concept') in ['S', 'C']
                
                logger.debug(f"  {concept.get('concept_name')} -> "
                            f"[{relation_id}] {related.get('concept_name')} (std={is_std})")
                
                if is_std:
                    # Standard concept - add directly
                    result.append({
                        'concept': related,
                        'is_original_standard': False,
                        'original_candidate': candidate['original_candidate'],
                        'original_non_standard': concept,
                        'relation_type': relation_id,
                        'elasticsearch_score': 0.0,
                        'search_type': candidate['search_type']
                    })
                else:
                    # Non-standard - apply Maps to
                    std_concepts = self._get_standard_via_maps_to(related_id)
                    for std in std_concepts:
                        logger.debug(f"    -> [Maps to] {std.get('concept_name')}")
                        result.append({
                            'concept': std,
                            'is_original_standard': False,
                            'original_candidate': candidate['original_candidate'],
                            'original_non_standard': related,
                            'relation_type': 'Maps to',
                            'elasticsearch_score': 0.0,
                            'search_type': candidate['search_type']
                        })
        
        return result
    
    def _second_transform(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        2차 변환: relationship 변환만 (Maps to 없음).
        
        - 1차 변환 결과에 relationship_id에 해당하는 관계가 있으면 변환
        """
        result = []
        processed_ids: Set[str] = set()
        
        for candidate in candidates:
            concept = candidate['concept']
            concept_id = str(concept.get('concept_id', ''))
            
            # 원본 컨셉 추가
            result.append(candidate)
            
            # 이미 처리한 concept_id는 스킵 (중복 relationship 조회 방지)
            if concept_id in processed_ids:
                continue
            processed_ids.add(concept_id)
            
            # Relationship 변환 시도
            related_concepts = self._get_related_concepts_with_relation(concept_id)
            
            for related, relation_id in related_concepts:
                logger.debug(f"  {concept.get('concept_name')} -> "
                            f"[{relation_id}] {related.get('concept_name')}")
                
                result.append({
                    'concept': related,
                    'is_original_standard': False,
                    'original_candidate': candidate['original_candidate'],
                    'original_non_standard': concept,
                    'relation_type': relation_id,
                    'elasticsearch_score': 0.0,
                    'search_type': candidate['search_type']
                })
        
        return result
    
    def _third_transform(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        3차 변환: non-std면 Maps to.
        
        - 2차 변환 결과 중 non-std면 Maps to로 std 변환
        """
        result = []
        
        for candidate in candidates:
            concept = candidate['concept']
            concept_id = str(concept.get('concept_id', ''))
            is_std = concept.get('standard_concept') in ['S', 'C']
            
            if is_std:
                # Standard - add directly
                result.append(candidate)
            else:
                # Non-standard - apply Maps to
                std_concepts = self._get_standard_via_maps_to(concept_id)
                
                if std_concepts:
                    for std in std_concepts:
                        logger.debug(f"  [3rd] {concept.get('concept_name')} -> "
                                    f"[Maps to] {std.get('concept_name')}")
                        result.append({
                            'concept': std,
                            'is_original_standard': False,
                            'original_candidate': candidate['original_candidate'],
                            'original_non_standard': concept,
                            'relation_type': 'Maps to',
                            'elasticsearch_score': 0.0,
                            'search_type': candidate['search_type']
                        })
                else:
                    # No Maps to found - keep original if needed
                    logger.debug(f"  [3rd] {concept.get('concept_name')} - no Maps to found")
        
        return result
    
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
                            {"terms": {"relationship_id.keyword": self.TRANSFORM_RELATIONSHIP_IDS}}
                        ]
                    }
                },
                "size": 20
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
