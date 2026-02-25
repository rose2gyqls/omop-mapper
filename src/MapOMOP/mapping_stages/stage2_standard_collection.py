"""
Stage 2: Standard Concept Collection

Converts concepts using relationship transformations and "Maps to" relationships.
Process (2 rounds of [relation transform → Maps to]):
1. 1차 라운드: Stage 1 후보군 → TRANSFORM_RELATIONSHIP_IDS 관계 변환
   → 변환된/미변환 후보 모두 non-std면 Maps to 적용
   → std 결과들만 2차로 넘김
2. 2차 라운드: std 결과들 → TRANSFORM_RELATIONSHIP_IDS 관계 변환
   → 변환된/미변환 후보 모두 non-std면 Maps to 적용
3. 1차+2차 결과 모두 합쳐서 중복 제거
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
        1. 1차 라운드: 관계 변환 → 변환/미변환 모두 Maps to → std만 2차로
        2. 2차 라운드: std 결과에 대해 같은 프로세스 반복
        3. 전체 결과 중복 제거
        
        Args:
            stage1_candidates: Candidates from Stage 1
            domain_id: Domain ID for logging
            
        Returns:
            Deduplicated standard candidates
        """
        logger.info("Stage 2: Standard Concept Collection")
        
        initial_candidates = self._prepare_initial_candidates(stage1_candidates)
        
        # 1차 라운드: 관계 변환 → Maps to
        first_round_all = self._apply_transform_and_maps_to(initial_candidates)
        first_round_dedup = deduplicate_by_concept(
            first_round_all,
            get_concept=lambda c: c.get('concept', {}),
            get_score=lambda c: c.get('elasticsearch_score', 0.0)
        )
        
        # 2차 라운드: std 결과만 2차 입력으로, 같은 프로세스
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
        
        # 1차 + 2차 전체 합쳐서 최종 중복 제거 후 stage3로
        all_results = first_round_dedup + second_round_dedup
        deduplicated = deduplicate_by_concept(
            all_results,
            get_concept=lambda c: c.get('concept', {}),
            get_score=lambda c: c.get('elasticsearch_score', 0.0)
        )
        
        self._log_candidates_detail("1차 라운드 결과", first_round_dedup)
        self._log_candidates_detail("2차 라운드 결과", second_round_dedup)
        self._log_candidates_detail("중복 제거 결과", deduplicated)
        
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
        """변환 결과 로깅: concept_name (concept_id) [relation_type] ← from 원본 (변환 관계 파악 가능)"""
        logger.info(f"  [{title}]")
        if not candidates:
            logger.info("    (없음)")
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
        [관계 변환 → Maps to] 한 라운드 적용.
        
        - TRANSFORM_RELATIONSHIP_IDS로 지정된 관계로 변환
        - 변환된 후보군 + 변환되지 않은 원본 모두에 대해:
          std면 그대로 추가, non-std면 Maps to로 std 찾아서 추가
        """
        result = []
        
        for candidate in candidates:
            concept = candidate['concept']
            concept_id = str(concept.get('concept_id', ''))
            
            # 1) 원본: std면 그대로, non-std면 Maps to
            self._add_candidate_or_maps_to(
                candidate, concept, candidate.get('original_non_standard'), 'original', result
            )
            
            # 2) Relationship 변환
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
        std면 그대로 추가, non-std면 Maps to로 std 찾아서 추가.
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
        """std(standard_concept in S/C)인 결과만 반환."""
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
    
