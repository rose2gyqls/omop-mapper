"""
Entity Mapping API Module

Main API for mapping medical entities to OMOP CDM standard concepts.
Uses a 3-stage pipeline: Candidate Retrieval → Standard Collection → Hybrid Scoring.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .elasticsearch_client import ElasticsearchClient
from .mapping_stages import (
    Stage1CandidateRetrieval,
    Stage2StandardCollection,
    Stage3HybridScoring,
    ScoringMode
)
from .mapping_validation import MappingValidator

# Optional dependencies
try:
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    HAS_SAPBERT = True
except ImportError:
    HAS_SAPBERT = False

logger = logging.getLogger(__name__)


class DomainID(Enum):
    """OMOP CDM domain identifiers."""
    PROCEDURE = "Procedure"
    CONDITION = "Condition"
    DRUG = "Drug"
    OBSERVATION = "Observation"
    MEASUREMENT = "Measurement"
    THRESHOLD = "Threshold"
    DEMOGRAPHICS = "Demographics"
    PERIOD = "Period"
    PROVIDER = "Provider"
    DEVICE = "Device"


@dataclass
class EntityInput:
    """Input entity for mapping."""
    entity_name: str
    domain_id: Optional[DomainID] = None
    vocabulary_id: Optional[str] = None


@dataclass
class MappingResult:
    """Mapping result data."""
    source_entity: EntityInput
    mapped_concept_id: str
    mapped_concept_name: str
    domain_id: str
    vocabulary_id: str
    concept_class_id: str
    standard_concept: str
    concept_code: str
    concept_embedding: List[float] = None
    valid_start_date: Optional[str] = None
    valid_end_date: Optional[str] = None
    invalid_reason: Optional[str] = None
    mapping_score: float = 0.0
    mapping_confidence: str = "low"
    mapping_method: str = "unknown"
    alternative_concepts: List[Dict[str, Any]] = field(default_factory=list)


class EntityMappingAPI:
    """
    3-stage entity mapping API.
    
    Stages:
        1. Candidate Retrieval: Lexical + Semantic + Combined search
        2. Standard Collection: Convert non-standard to standard concepts
        3. Hybrid Scoring: LLM or embedding-based final ranking
    """
    
    def __init__(
        self,
        es_client: Optional[ElasticsearchClient] = None,
        confidence_threshold: float = 0.5,
        scoring_mode: str = ScoringMode.LLM,
        include_non_std_info: bool = False
    ):
        """
        Initialize mapping API.
        
        Args:
            es_client: Elasticsearch client instance
            confidence_threshold: Minimum confidence threshold
            scoring_mode: Scoring mode
                - 'llm': LLM without score (default)
                - 'llm_with_score': LLM with semantic score in prompt
                - 'semantic': Semantic similarity only
            include_non_std_info: Include non-std concept info in LLM prompt
        """
        self.es_client = es_client or ElasticsearchClient.create_default()
        self.confidence_threshold = confidence_threshold
        self.scoring_mode = scoring_mode
        self.include_non_std_info = include_non_std_info
        
        # SapBERT model (lazy loading)
        self._sapbert_model = None
        self._sapbert_tokenizer = None
        self._sapbert_device = None
        
        # Stage modules
        self.stage1 = Stage1CandidateRetrieval(
            es_client=self.es_client,
            has_sapbert=HAS_SAPBERT
        )
        
        self.stage2 = Stage2StandardCollection(es_client=self.es_client)
        self.stage3 = None  # Initialized after SapBERT loading
        
        # Validation module (uses LLM client from environment config)
        self.validator = MappingValidator(es_client=self.es_client)
        
        # Debug variables
        self._last_stage1_candidates = []
        self._last_stage2_candidates = []
        self._last_rerank_candidates = []
    
    def map_entity(self, entity_input: EntityInput) -> Optional[List[MappingResult]]:
        """
        Map entity to OMOP CDM using 3-stage pipeline.
        
        Args:
            entity_input: Entity to map
            
        Returns:
            List of MappingResult for each domain (sorted by score)
        """
        # Reset debug variables at the start of each mapping
        self._last_stage1_candidates = []
        self._last_stage2_candidates = []
        self._last_rerank_candidates = []
        
        try:
            entity_name = entity_input.entity_name
            input_domain = entity_input.domain_id
            
            # Determine target domains
            if input_domain is None:
                target_domains = [
                    DomainID.DRUG, DomainID.OBSERVATION, DomainID.PROCEDURE,
                    DomainID.CONDITION, DomainID.MEASUREMENT, DomainID.DEVICE
                ]
                logger.info("=" * 80)
                logger.info(f"Starting multi-domain mapping: {entity_name}")
                logger.info(f"Target domains: {[d.value for d in target_domains]}")
                logger.info("=" * 80)
            else:
                target_domains = [input_domain]
                logger.info("=" * 80)
                logger.info(f"Starting single-domain mapping: {entity_name}")
                logger.info(f"Target domain: {input_domain.value}")
                logger.info("=" * 80)
            
            # Initialize SapBERT model
            if HAS_SAPBERT and self._sapbert_model is None:
                self._initialize_sapbert_model()
            
            # Initialize Stage 3 (uses LLM client from environment config)
            if self.stage3 is None:
                self.stage3 = Stage3HybridScoring(
                    sapbert_model=self._sapbert_model,
                    sapbert_tokenizer=self._sapbert_tokenizer,
                    sapbert_device=self._sapbert_device,
                    es_client=self.es_client,
                    scoring_mode=self.scoring_mode,
                    include_non_std_info=self.include_non_std_info
                )
            
            # Generate entity embedding
            entity_embedding = None
            if HAS_SAPBERT and self._sapbert_model is not None:
                entity_embedding = self._get_embedding(entity_name)
                if entity_embedding is not None:
                    logger.info("Entity embedding generated successfully")
            
            # Process each domain
            all_results = []
            domain_candidates = {}
            result_to_domain = {}
            
            for domain in target_domains:
                domain_result, domain_stages = self._map_entity_for_domain(
                    entity_name=entity_name,
                    domain_id=domain,
                    entity_embedding=entity_embedding,
                    entity_input=entity_input
                )
                
                domain_str = str(domain.value)
                
                # Always store candidates (even if domain_result is None)
                if 'candidates' in domain_stages:
                    domain_candidates[domain_str] = domain_stages['candidates']
                
                if domain_result:
                    all_results.append(domain_result)
                    result_to_domain[id(domain_result)] = domain_str
            
            logger.info("=" * 80)
            logger.info(f"Mapping complete: {len(all_results)} domain(s) returned results")
            logger.info("=" * 80)
            
            # Set debug variables
            if all_results:
                # Success case: use best result's candidates
                for idx, result in enumerate(all_results, 1):
                    logger.info(f"  {idx}. [{result.domain_id}] {result.mapped_concept_name} "
                               f"(score: {result.mapping_score:.4f})")
                
                best = max(all_results, key=lambda x: x.mapping_score)
                logger.info(f"Best match: [{best.domain_id}] {best.mapped_concept_name} "
                           f"({best.mapping_score:.4f})")
                
                best_domain = result_to_domain.get(id(best))
                if best_domain and best_domain in domain_candidates:
                    self._last_stage1_candidates = domain_candidates[best_domain].get('stage1', [])
                    self._last_stage2_candidates = domain_candidates[best_domain].get('stage2', [])
                    self._last_rerank_candidates = domain_candidates[best_domain].get('stage3', [])
            else:
                # Failure case: use domain with most candidate info (avoid overwriting with empty)
                if domain_candidates:
                    best_fail_domain = self._select_domain_with_most_candidates(domain_candidates)
                    self._last_stage1_candidates = domain_candidates[best_fail_domain].get('stage1', [])
                    self._last_stage2_candidates = domain_candidates[best_fail_domain].get('stage2', [])
                    self._last_rerank_candidates = domain_candidates[best_fail_domain].get('stage3', [])
                    logger.info(f"Mapping failed. Stage candidates recorded for debugging.")
            
            return all_results if all_results else None
            
        except Exception as e:
            logger.error(f"Entity mapping error: {e}", exc_info=True)
            return None
    
    def _select_domain_with_most_candidates(
        self, domain_candidates: Dict[str, Dict]
    ) -> str:
        """
        Select domain with most candidate info when all domains failed.
        Prefer: stage3 > stage2 > stage1 (more stages = more debug info).
        """
        def score_domain(candidates: Dict) -> int:
            s1 = len(candidates.get('stage1', []))
            s2 = len(candidates.get('stage2', []))
            s3 = len(candidates.get('stage3', []))
            return s3 * 10000 + s2 * 100 + s1
        
        best_domain = list(domain_candidates.keys())[0]
        best_score = score_domain(domain_candidates[best_domain])
        
        for domain, candidates in domain_candidates.items():
            s = score_domain(candidates)
            if s > best_score:
                best_score = s
                best_domain = domain
        
        return best_domain
    
    def _map_entity_for_domain(
        self,
        entity_name: str,
        domain_id: DomainID,
        entity_embedding,
        entity_input: EntityInput
    ) -> tuple[Optional[MappingResult], Dict[str, Any]]:
        """Map entity for a specific domain."""
        try:
            domain_str = domain_id.value
            
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing domain: {domain_str}")
            logger.info('=' * 60)
            
            stage_results = {
                'search_domain': domain_str,
                'result_domain': None,
                'candidates': {}
            }
            
            # Stage 1: Candidate Retrieval
            es_index = getattr(self.es_client, 'concept_index', 'concept')
            stage1_candidates = self.stage1.retrieve_candidates(
                entity_name=entity_name,
                domain_id=domain_str,
                entity_embedding=entity_embedding,
                es_index=es_index
            )
            
            if not stage1_candidates:
                logger.info(f"[{domain_str}] Stage 1: No candidates found")
                stage_results['failure_reason'] = 'stage1_no_candidates'
                stage_results['candidates'] = {
                    'stage1': [],
                    'stage2': [],
                    'stage3': []
                }
                return None, stage_results
            
            # Stage 2: Standard Collection
            stage2_candidates = self.stage2.collect_standard_candidates(
                stage1_candidates=stage1_candidates,
                domain_id=domain_str
            )
            
            if not stage2_candidates:
                logger.info(f"[{domain_str}] Stage 2: No standard candidates found")
                stage_results['failure_reason'] = 'stage2_no_candidates'
                stage_results['candidates'] = {
                    'stage1': [self._format_stage1_candidate(h) for h in stage1_candidates],
                    'stage2': [],
                    'stage3': []
                }
                return None, stage_results
            
            # Stage 3: Hybrid Scoring
            stage3_candidates = self.stage3.calculate_hybrid_scores(
                entity_name=entity_name,
                stage2_candidates=stage2_candidates,
                stage1_candidates=stage1_candidates,
                entity_embedding=entity_embedding
            )
            
            if not stage3_candidates:
                logger.info(f"[{domain_str}] Stage 3: Scoring failed")
                stage_results['failure_reason'] = 'stage3_scoring_failed'
                stage_results['candidates'] = {
                    'stage1': [self._format_stage1_candidate(h) for h in stage1_candidates],
                    'stage2': [self._format_stage2_candidate(c) for c in stage2_candidates],
                    'stage3': []
                }
                return None, stage_results
            
            # Create mapping result
            domain_entity_input = EntityInput(
                entity_name=entity_input.entity_name,
                domain_id=domain_id,
                vocabulary_id=entity_input.vocabulary_id
            )
            
            # Validation: try top 3 candidates by score (highest first)
            logger.info(f"\n{'=' * 60}")
            logger.info("Validation step (top 3 by score)")
            logger.info('=' * 60)
            
            # Keep original stage3_candidates for logging (sorted by LLM score)
            final_stage3_candidates = stage3_candidates
            
            MAX_VALIDATION_ATTEMPTS = 3
            validation_candidates = stage3_candidates[:MAX_VALIDATION_ATTEMPTS]
            
            validated_candidate = None
            validated_idx = None
            
            for idx, candidate in enumerate(validation_candidates):
                concept = candidate.get('concept', {})
                concept_id = str(concept.get('concept_id', ''))
                concept_name = concept.get('concept_name', '')
                candidate_score = candidate.get('final_score', 0.0)
                
                logger.info(f"  [{idx + 1}/{len(validation_candidates)}] Validating: "
                           f"{concept_name} (ID: {concept_id}, score: {candidate_score:.2f})")
                
                is_valid = self.validator.validate_mapping(
                    entity_name=entity_name,
                    concept_id=concept_id,
                    concept_name=concept_name,
                    synonyms=None
                )
                
                if is_valid:
                    logger.info(f"  Validated: {concept_name}")
                    validated_candidate = candidate
                    validated_idx = idx
                    break
                else:
                    logger.info(f"  Failed: {concept_name}")
            
            if validated_candidate is None:
                logger.error(f"[{domain_str}] Top {len(validation_candidates)} candidates all failed validation")
                stage_results['validation_status'] = 'failed'
                stage_results['candidates'] = {
                    'stage1': [self._format_stage1_candidate(h) for h in stage1_candidates],
                    'stage2': [self._format_stage2_candidate(c) for c in stage2_candidates],
                    'stage3': [self._format_stage3_candidate(c) for c in stage3_candidates]
                }
                return None, stage_results
            
            # Build final result from validated candidate
            if validated_idx == 0:
                # Top scored candidate passed validation
                mapping_result = self._create_final_result(domain_entity_input, stage3_candidates)
                stage_results['validation_status'] = 'validated'
                stage_results['validation_changed_result'] = False
                logger.info(f"[{domain_str}] Top candidate validated: {validated_candidate['concept'].get('concept_name')}")
            else:
                # Alternative candidate passed validation - reorder
                llm_top_name = stage3_candidates[0]['concept'].get('concept_name')
                llm_top_id = stage3_candidates[0]['concept'].get('concept_id')
                validated_name = validated_candidate['concept'].get('concept_name')
                
                logger.info(f"  [!] LLM top pick was: {llm_top_name} (ID: {llm_top_id})")
                logger.info(f"  [!] Changed to: {validated_name} due to validation")
                
                reordered = [stage3_candidates[validated_idx]] + \
                           [x for i, x in enumerate(stage3_candidates) if i != validated_idx]
                mapping_result = self._create_final_result(domain_entity_input, reordered)
                stage_results['validation_status'] = 'validated_alternative'
                stage_results['llm_top_pick'] = {
                    'concept_id': str(llm_top_id),
                    'concept_name': llm_top_name
                }
                stage_results['validation_changed_result'] = True
            
            stage_results['result_domain'] = mapping_result.domain_id
            
            # Store candidates for debugging (original LLM ranking order)
            stage_results['candidates'] = {
                'stage1': [self._format_stage1_candidate(h) for h in stage1_candidates],
                'stage2': [self._format_stage2_candidate(c) for c in stage2_candidates],
                'stage3': [self._format_stage3_candidate(c) for c in final_stage3_candidates]
            }
            
            logger.info(f"\n[{domain_str}] Mapping complete: {mapping_result.mapped_concept_name}")
            logger.info(f"  Score: {mapping_result.mapping_score:.4f} | "
                       f"Confidence: {mapping_result.mapping_confidence}")
            
            return mapping_result, stage_results
            
        except Exception as e:
            logger.error(f"[{domain_str}] Mapping error: {e}", exc_info=True)
            return None, {}
    
    def _create_final_result(
        self,
        entity_input: EntityInput,
        sorted_candidates: List[Dict[str, Any]]
    ) -> MappingResult:
        """Create final mapping result from sorted candidates."""
        best = sorted_candidates[0]
        alternatives = sorted_candidates[1:4]
        
        concept = best['concept']
        score = best.get('final_score', 0.0)
        
        # Extract alternative concepts
        alt_concepts = []
        for alt in alternatives:
            if 'concept' in alt:
                alt_c = alt['concept']
                alt_concepts.append({
                    'concept_id': str(alt_c.get('concept_id', '')),
                    'concept_name': alt_c.get('concept_name', ''),
                    'vocabulary_id': alt_c.get('vocabulary_id', ''),
                    'score': alt.get('final_score', 0)
                })
        
        method = "direct_standard" if best.get('is_original_standard', True) else "non_standard_to_standard"
        
        return MappingResult(
            source_entity=entity_input,
            mapped_concept_id=str(concept.get('concept_id', '')),
            mapped_concept_name=concept.get('concept_name', ''),
            domain_id=concept.get('domain_id', ''),
            vocabulary_id=concept.get('vocabulary_id', ''),
            concept_class_id=concept.get('concept_class_id', ''),
            standard_concept=concept.get('standard_concept', ''),
            concept_code=concept.get('concept_code', ''),
            valid_start_date=concept.get('valid_start_date'),
            valid_end_date=concept.get('valid_end_date'),
            invalid_reason=concept.get('invalid_reason'),
            concept_embedding=concept.get('concept_embedding'),
            mapping_score=score,
            mapping_confidence=self._get_confidence(score),
            mapping_method=method,
            alternative_concepts=alt_concepts
        )
    
    def _get_confidence(self, score: float) -> str:
        """Determine confidence level from score."""
        if score >= 0.95:
            return "very_high"
        elif score >= 0.85:
            return "high"
        elif score >= 0.70:
            return "medium"
        elif score >= 0.50:
            return "low"
        else:
            return "very_low"
    
    def _format_stage1_candidate(self, hit: Dict) -> Dict:
        """Format Stage 1 candidate for storage."""
        src = hit['_source']
        # 정규화 점수 사용 (0~1), 없으면 raw score 사용
        normalized_score = hit.get('_score_normalized') or hit['_score']
        return {
            'concept_id': str(src.get('concept_id', '')),
            'concept_name': src.get('concept_name', ''),
            'domain_id': src.get('domain_id', ''),
            'vocabulary_id': src.get('vocabulary_id', ''),
            'standard_concept': src.get('standard_concept', ''),
            'elasticsearch_score': normalized_score,
            'search_type': hit.get('_search_type', 'unknown')
        }
    
    def _format_stage2_candidate(self, c: Dict) -> Dict:
        """Format Stage 2 candidate for storage."""
        concept = c['concept']
        result = {
            'concept_id': str(concept.get('concept_id', '')),
            'concept_name': concept.get('concept_name', ''),
            'domain_id': concept.get('domain_id', ''),
            'vocabulary_id': concept.get('vocabulary_id', ''),
            'standard_concept': concept.get('standard_concept', ''),
            'is_original_standard': c.get('is_original_standard', True),
            'search_type': c.get('search_type', 'unknown'),
            'relation_type': c.get('relation_type', 'original'),
            'elasticsearch_score': c.get('elasticsearch_score', 0.0)
        }
        # Include original non-standard info if present
        if not c.get('is_original_standard', True) and 'original_non_standard' in c:
            non_std = c['original_non_standard']
            result['original_non_standard'] = {
                'concept_id': str(non_std.get('concept_id', '')),
                'concept_name': non_std.get('concept_name', '')
            }
        return result
    
    def _format_stage3_candidate(self, c: Dict) -> Dict:
        """Format Stage 3 candidate for storage."""
        concept = c['concept']
        result = {
            'concept_id': str(concept.get('concept_id', '')),
            'concept_name': concept.get('concept_name', ''),
            'domain_id': concept.get('domain_id', ''),
            'vocabulary_id': concept.get('vocabulary_id', ''),
            'standard_concept': concept.get('standard_concept', ''),
            'is_original_standard': c.get('is_original_standard', True),
            'llm_score': c.get('llm_score'),
            'llm_rank': c.get('llm_rank'),
            'llm_reasoning': c.get('llm_reasoning', ''),
            'final_score': c.get('final_score', 0.0),
            'search_type': c.get('search_type', 'unknown'),
            'semantic_similarity': c.get('semantic_similarity'),
            'text_similarity': c.get('text_similarity')
        }
        # Include original non-standard info if present
        if not c.get('is_original_standard', True) and 'original_non_standard' in c:
            non_std = c['original_non_standard']
            result['original_non_standard'] = {
                'concept_id': str(non_std.get('concept_id', '')),
                'concept_name': non_std.get('concept_name', '')
            }
        return result
    
    def _get_embedding(self, text: str):
        """Generate SapBERT embedding for text (with dimension reduction)."""
        try:
            if self._sapbert_model is None:
                self._initialize_sapbert_model()
            
            if self._sapbert_model is None:
                return None
            
            text = text.lower().strip()
            
            inputs = self._sapbert_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=25
            )
            inputs = {k: v.to(self._sapbert_device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._sapbert_model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            embedding = embedding.flatten()
            
            # Apply same dimension reduction as indexing (768 → 128)
            from .utils import reduce_embedding_dim
            embedding = reduce_embedding_dim(embedding)
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None
    
    def _initialize_sapbert_model(self):
        """Initialize SapBERT model (lazy loading)."""
        if not HAS_SAPBERT:
            logger.warning("SapBERT dependencies not installed")
            return
        
        try:
            model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
            logger.info(f"Loading SapBERT model: {model_name}")
            
            self._sapbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._sapbert_model = AutoModel.from_pretrained(model_name)
            
            if torch.cuda.is_available():
                self._sapbert_device = torch.device('cuda')
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self._sapbert_device = torch.device('cpu')
                logger.info("Using CPU")
            
            self._sapbert_model.to(self._sapbert_device)
            self._sapbert_model.eval()
            
            logger.info("SapBERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"SapBERT model loading failed: {e}")
            self._sapbert_model = None
            self._sapbert_tokenizer = None
            self._sapbert_device = None
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        return {
            "api_status": "healthy",
            "elasticsearch_status": self.es_client.health_check(),
            "confidence_threshold": self.confidence_threshold
        }
