"""
Stage 3: Hybrid Scoring

Final scoring and ranking of candidates using multiple strategies:
- llm: LLM-based evaluation without semantic scores (default)
- llm_with_score: LLM-based evaluation with semantic scores in prompt
- semantic: SapBERT cosine similarity only

Supports multiple LLM providers via LLMClient:
- OpenAI (gpt-4o-mini, etc.)
- SNUH Hari (snuh/hari-q3-14b)
- Google Gemma (google/gemma-3-12b-it)
"""

import json
import logging
from typing import Any, Dict, List, Optional

from ..utils import sigmoid_normalize
from ..llm_client import LLMClient, get_llm_client, LLMProvider

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


# =============================================================================
# Prompt Templates (easily customizable)
# =============================================================================

SYSTEM_PROMPT = """You are given an entity and several candidate OMOP CDM concepts. Your task is to score EACH candidate on a scale of 0-5 based on how well it matches the entity."""

USER_PROMPT_TEMPLATE = """
If a candidate is SEMANTICALLY EQUIVALENT(i.e., it has the same clinical meaning, even if the wording is different)
, you MUST select it.

If no semantically equivalent candidate exists, you MAY select a MORE GENERAL
(higher-level) concept ONLY IF it fully preserves the clinical meaning and does
NOT introduce any additional or different meaning.

IMPORTANT:
- Semantic equivalence is based on clinical meaning, NOT on string similarity.
- Do NOT guess.
- Choosing a wrong concept with additional or different meaning is worse than choosing NONE.
- If generalization would change the meaning, select NONE.
- CRITICAL DEFINITION OF ADDITIONAL MEANING: Adding a specific body site, anatomical structure, severity, or underlying cause
that is NOT present in the original entity is considered adding "new meaning" and makes the candidate an invalid sub-concept.

Decision process (follow strictly):
1. ELIMINATE any candidate that changes the core clinical meaning OR adds specific details (like body parts or causes) not found in the original entity.
2. Among remaining candidates:
   a) Prefer a SEMANTICALLY EQUIVALENT concept.
   b) If none exists, prefer the MOST APPROPRIATE higher-level concept
      that fully contains the entity meaning without adding new meaning.
3. NEVER select a more specific sub-concept.

**Entity**: {entity_name}

**Candidates**:
{candidates_json}
{score_hint}

**Scoring Guide** (use decimal scores from 0.0 to 5.0):
- 5.0: EXACT MATCH or SEMANTICALLY EQUIVALENT (same clinical meaning)
- 4.0~4.9: Very close match (minor wording difference, clinically identical)
- 3.0~3.9: Acceptable generalization (broader concept that preserves meaning)
- 2.0~2.9: Partial match (related but loses some meaning)
- 1.0~1.9: Weak match (loosely related)
- 0.0~0.9: No match OR UNACCEPTABLE SUB-CONCEPT (different meaning or adds new meaning)

CRITICAL: Every candidate MUST have a UNIQUE score. No two candidates may share the same score. Use decimal precision (e.g., 4.8, 4.3, 3.5) to differentiate candidates.

**Output Format** (JSON only):
{{
  "rankings": [
    {{"concept_id": "ID", "score": 0.0-5.0, "reasoning": "reason"}},
    ...
  ]
}}
"""

SCORE_HINT_TEMPLATE = """
**Semantic Similarity Info**:
- semantic_similarity: SapBERT embedding cosine similarity (0.0-1.0)
  - Higher = more semantically similar
  - Use as reference, but prioritize medical accuracy
"""


class ScoringMode:
    """Available scoring modes."""
    LLM = "llm"                    # LLM without score (default)
    LLM_WITH_SCORE = "llm_with_score"  # LLM with semantic score
    SEMANTIC = "semantic"          # Semantic similarity only


class Stage3HybridScoring:
    """Stage 3: Hybrid/LLM-based candidate scoring."""
    
    # Prompt templates (can be overridden)
    SYSTEM_PROMPT = SYSTEM_PROMPT
    USER_PROMPT_TEMPLATE = USER_PROMPT_TEMPLATE
    SCORE_HINT_TEMPLATE = SCORE_HINT_TEMPLATE
    
    # Default LLM hyperparameters
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_TOP_P = 1.0
    
    def __init__(
        self,
        sapbert_model=None,
        sapbert_tokenizer=None,
        sapbert_device=None,
        es_client=None,
        llm_client: Optional[LLMClient] = None,
        scoring_mode: str = ScoringMode.LLM,
        include_non_std_info: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ):
        """
        Initialize Stage 3.
        
        Args:
            sapbert_model: SapBERT model (for semantic mode)
            sapbert_tokenizer: SapBERT tokenizer
            sapbert_device: SapBERT device
            es_client: Elasticsearch client
            llm_client: LLM client instance (uses default if None)
            scoring_mode: Scoring mode
                - 'llm': LLM without score (default)
                - 'llm_with_score': LLM with semantic score in prompt
                - 'semantic': Semantic similarity only
            include_non_std_info: Include original non-std concept info in LLM prompt
            temperature: LLM temperature (0.0-2.0, default from env or 0.3)
            top_p: LLM top_p / nucleus sampling (0.0-1.0, default from env or 1.0)
        """
        self.es_client = es_client
        self.scoring_mode = scoring_mode.lower()
        self.include_non_std_info = include_non_std_info
        
        # SapBERT settings (for semantic scoring)
        self.sapbert_model = sapbert_model
        self.sapbert_tokenizer = sapbert_tokenizer
        self.sapbert_device = sapbert_device
        
        # LLM client (supports OpenAI, Hari, Gemma)
        self.llm_client = llm_client
        self.temperature = temperature
        self.top_p = top_p
        
        # Initialize based on mode
        self._initialize_mode()
    
    def _initialize_mode(self):
        """Initialize based on scoring mode."""
        if self.scoring_mode in [ScoringMode.LLM, ScoringMode.LLM_WITH_SCORE]:
            # Use provided client or get default
            if self.llm_client is None:
                self.llm_client = get_llm_client()
            
            if self.llm_client.is_initialized:
                mode_desc = "with score" if self.scoring_mode == ScoringMode.LLM_WITH_SCORE else "without score"
                llm_info = self.llm_client.get_info()
                logger.info(
                    f"Stage 3 initialized (LLM mode {mode_desc}, "
                    f"provider: {llm_info['provider']}, model: {llm_info['model']})"
                )
            else:
                logger.error("LLM client not initialized")
        elif self.scoring_mode == ScoringMode.SEMANTIC:
            logger.info("Stage 3 initialized (Semantic mode)")
        else:
            logger.warning(f"Unknown scoring mode: {self.scoring_mode}, defaulting to 'llm'")
            self.scoring_mode = ScoringMode.LLM
    
    @property
    def include_scores_in_prompt(self) -> bool:
        """Whether to include semantic scores in LLM prompt."""
        return self.scoring_mode == ScoringMode.LLM_WITH_SCORE
    
    def calculate_hybrid_scores(
        self,
        entity_name: str,
        stage2_candidates: List[Dict[str, Any]],
        stage1_candidates: Optional[List[Dict[str, Any]]] = None,
        entity_embedding: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Calculate final scores and rank candidates.
        
        Args:
            entity_name: Entity name
            stage2_candidates: Candidates from Stage 2
            stage1_candidates: Stage 1 candidates (unused, for compatibility)
            entity_embedding: Entity SapBERT embedding
            
        Returns:
            Sorted candidates with final scores
        """
        mode_display_names = {
            ScoringMode.LLM: 'LLM (w/o score)',
            ScoringMode.LLM_WITH_SCORE: 'LLM (with score)',
            ScoringMode.SEMANTIC: 'Semantic'
        }
        
        if not stage2_candidates:
            logger.warning("No candidates to score")
            return []
        
        if self.scoring_mode in [ScoringMode.LLM, ScoringMode.LLM_WITH_SCORE]:
            return self._score_llm(entity_name, stage2_candidates, entity_embedding)
        elif self.scoring_mode == ScoringMode.SEMANTIC:
            return self._score_semantic(entity_name, stage2_candidates, entity_embedding)
        else:
            logger.error(f"Unknown scoring mode: {self.scoring_mode}")
            return []
    
    def _score_semantic(
        self,
        entity_name: str,
        candidates: List[Dict[str, Any]],
        entity_embedding: Optional[Any]
    ) -> List[Dict[str, Any]]:
        """Score using semantic similarity only."""
        if entity_embedding is None:
            logger.warning("No entity embedding for semantic scoring")
            return []
        
        results = []
        
        for candidate in candidates:
            concept = candidate['concept']
            concept_emb = concept.get('concept_embedding')
            semantic_sim = self._compute_cosine(entity_embedding, concept_emb) or 0.0
            
            # Elasticsearch score (sigmoid normalized)
            es_score = candidate.get('elasticsearch_score', 0.0)
            es_score_normalized = sigmoid_normalize(es_score, center=3.0)
            
            results.append({
                'concept': concept,
                'is_original_standard': candidate.get('is_original_standard', True),
                'original_candidate': candidate.get('original_candidate', {}),
                'original_non_standard': candidate.get('original_non_standard'),
                'elasticsearch_score': es_score,
                'elasticsearch_score_normalized': es_score_normalized,
                'search_type': candidate.get('search_type', 'unknown'),
                'semantic_similarity': semantic_sim,
                'final_score': semantic_sim
            })
        
        sorted_results = sorted(results, key=lambda x: x['final_score'], reverse=True)
        
        self._log_results("Semantic", sorted_results)
        return sorted_results
    
    def _score_llm(
        self,
        entity_name: str,
        candidates: List[Dict[str, Any]],
        entity_embedding: Optional[Any]
    ) -> List[Dict[str, Any]]:
        """Score using LLM evaluation."""
        if not self.llm_client or not self.llm_client.is_initialized:
            logger.error("LLM client not initialized")
            return []
        
        # Prepare candidates with optional semantic scores
        results = []
        for candidate in candidates:
            concept = candidate['concept']
            
            # Elasticsearch score (sigmoid normalized)
            es_score = candidate.get('elasticsearch_score', 0.0)
            es_score_normalized = sigmoid_normalize(es_score, center=3.0)
            
            data = {
                'concept': concept,
                'is_original_standard': candidate.get('is_original_standard', True),
                'original_candidate': candidate.get('original_candidate', {}),
                'original_non_standard': candidate.get('original_non_standard'),
                'elasticsearch_score': es_score,
                'elasticsearch_score_normalized': es_score_normalized,
                'search_type': candidate.get('search_type', 'unknown')
            }
            
            # Add semantic similarity if mode requires it
            if self.include_scores_in_prompt and entity_embedding is not None:
                concept_emb = concept.get('concept_embedding')
                data['semantic_similarity'] = self._compute_cosine(entity_embedding, concept_emb) or 0.0
            
            results.append(data)
        
        # Get LLM scores
        try:
            llm_scores = self._call_llm(entity_name, results)
            
            if not llm_scores:
                logger.error("LLM scoring failed")
                return []
            
            # Apply scores
            for candidate in results:
                cid = str(candidate['concept'].get('concept_id', ''))
                if cid in llm_scores:
                    candidate['llm_score'] = llm_scores[cid]['score']
                    candidate['llm_rank'] = llm_scores[cid]['rank']
                    candidate['llm_reasoning'] = llm_scores[cid].get('reasoning', '')
                    candidate['final_score'] = candidate['llm_score']
                else:
                    candidate['llm_score'] = 0.0
                    candidate['llm_rank'] = 999
                    candidate['final_score'] = 0.0
            
            sorted_results = sorted(results, key=lambda x: x['llm_score'], reverse=True)
            
            self._log_results("LLM", sorted_results)
            return sorted_results
            
        except Exception as e:
            logger.error(f"LLM scoring failed: {e}")
            return []
    
    def _compute_cosine(self, emb1: Any, emb2: Any) -> Optional[float]:
        """Compute cosine similarity between embeddings."""
        if emb1 is None or emb2 is None or not HAS_NUMPY:
            return None
        
        try:
            # Convert to numpy arrays
            if isinstance(emb2, str):
                emb2 = np.array(json.loads(emb2))
            elif isinstance(emb2, list):
                emb2 = np.array(emb2)
            
            if isinstance(emb1, list):
                emb1 = np.array(emb1)
            
            if emb1 is None or emb2 is None:
                return None
            
            # Compute similarity
            sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
            return float((sim + 1) / 2)  # Normalize to 0-1
            
        except Exception as e:
            logger.debug(f"Cosine computation failed: {e}")
            return None
    
    def _call_llm(
        self,
        entity_name: str,
        candidates: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Call LLM API for scoring."""
        prompt = self._build_prompt(entity_name, candidates)
        
        try:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_client.chat_completion(
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=2048,
                json_mode=True
            )
            
            if response is None:
                return None
            
            return self._parse_llm_response(response, candidates)
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return None
    
    def _build_prompt(self, entity_name: str, candidates: List[Dict[str, Any]]) -> str:
        """Build LLM prompt from template."""
        candidates_info = []
        for i, c in enumerate(candidates, 1):
            concept = c['concept']
            concept_name = concept.get('concept_name', '')
            
            # If include_non_std_info is True and this is a non-std to std mapping,
            # show as "std_concept_name (non_std_concept_name)"
            if self.include_non_std_info and not c.get('is_original_standard', True):
                original_non_std = c.get('original_non_standard')
                if original_non_std:
                    non_std_name = original_non_std.get('concept_name', '')
                    if non_std_name and non_std_name != concept_name:
                        concept_name = f"{concept_name} ({non_std_name})"
            
            info = {
                'index': i,
                'concept_id': str(concept.get('concept_id', '')),
                'concept_name': concept_name,
                'domain_id': concept.get('domain_id', '')
            }
            
            # Add semantic similarity if mode requires it
            if self.include_scores_in_prompt and 'semantic_similarity' in c:
                info['semantic_similarity'] = round(c['semantic_similarity'], 4)
            
            candidates_info.append(info)
        
        # Build score hint
        score_hint = self.SCORE_HINT_TEMPLATE if self.include_scores_in_prompt else ""
        
        # Build final prompt from template
        return self.USER_PROMPT_TEMPLATE.format(
            entity_name=entity_name,
            candidates_json=json.dumps(candidates_info, ensure_ascii=False, indent=2),
            score_hint=score_hint
        )
    
    def _parse_llm_response(
        self,
        response_text: str,
        candidates: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Parse LLM response."""
        try:
            text = response_text.strip()
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()
            
            parsed = json.loads(text)
            result = {}
            
            rankings = parsed.get('rankings', [])
            
            # Sort by score descending and assign ranks
            rankings_sorted = sorted(rankings, key=lambda x: float(x.get('score', 0)), reverse=True)
            
            for rank, item in enumerate(rankings_sorted, 1):
                cid = str(item.get('concept_id', ''))
                if cid:
                    result[cid] = {
                        'score': float(item.get('score', 0.0)),
                        'rank': rank,
                        'reasoning': item.get('reasoning', '')
                    }
            
            # Ensure all candidates are in result
            for c in candidates:
                cid = str(c['concept'].get('concept_id', ''))
                if cid not in result:
                    result[cid] = {'score': 0.0, 'rank': 999, 'reasoning': 'Not ranked'}
            
            return result
            
        except Exception as e:
            logger.error(f"LLM response parsing failed: {e}")
            return {}
    
    def _log_results(self, mode: str, results: List[Dict[str, Any]]):
        """Log scoring results: concept_name (concept_id) score, reasoning."""
        logger.info("Stage 3: Scoring Results")
        for r in results[:15]:
            concept = r['concept']
            name = concept.get('concept_name', 'N/A')
            cid = concept.get('concept_id', 'N/A')
            score = r.get('final_score', 0.0)
            logger.info(f"  {name} ({cid}) {score:.4f}")
            reasoning = r.get('llm_reasoning', '')
            if reasoning:
                logger.info(f"    → {reasoning}")
