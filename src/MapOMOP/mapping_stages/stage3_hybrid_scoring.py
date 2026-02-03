"""
Stage 3: Hybrid Scoring

Final scoring and ranking of candidates using multiple strategies:
- LLM: OpenAI API-based evaluation
- Hybrid: Text similarity + Semantic similarity
- Semantic Only: SapBERT cosine similarity only
"""

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def sigmoid_normalize(score: float, center: float = 3.0, scale: float = 1.0) -> float:
    """
    Normalize ES score to 0-1 range using sigmoid.
    
    Args:
        score: Raw ES score (e.g., BM25 score)
        center: Score at which sigmoid returns 0.5
        scale: Steepness of the curve
        
    Returns:
        Normalized score between 0 and 1
    """
    return 1 / (1 + math.exp(-(score - center) / scale))

# Optional dependencies
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class Stage3HybridScoring:
    """Stage 3: Hybrid/LLM-based candidate scoring."""
    
    def __init__(
        self,
        sapbert_model=None,
        sapbert_tokenizer=None,
        sapbert_device=None,
        text_weight: float = 0.4,
        semantic_weight: float = 0.6,
        es_client=None,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
        scoring_mode: str = "llm",
        include_stage1_scores: bool = False,
        include_non_std_info: bool = False
    ):
        """
        Initialize Stage 3.
        
        Args:
            sapbert_model: SapBERT model (for hybrid mode)
            sapbert_tokenizer: SapBERT tokenizer
            sapbert_device: SapBERT device
            text_weight: Text similarity weight (default: 0.4)
            semantic_weight: Semantic similarity weight (default: 0.6)
            es_client: Elasticsearch client
            openai_api_key: OpenAI API key
            openai_model: OpenAI model name
            scoring_mode: 'llm', 'hybrid', or 'semantic_only'
            include_stage1_scores: Include scores in LLM prompt
            include_non_std_info: Include original non-std concept info in LLM prompt
        """
        self.es_client = es_client
        self.scoring_mode = scoring_mode.lower()
        self.include_stage1_scores = include_stage1_scores
        self.include_non_std_info = include_non_std_info
        
        # Hybrid mode settings
        self.sapbert_model = sapbert_model
        self.sapbert_tokenizer = sapbert_tokenizer
        self.sapbert_device = sapbert_device
        self.text_weight = text_weight
        self.semantic_weight = semantic_weight
        
        # LLM settings
        self.openai_client = None
        self.openai_model = openai_model
        
        if self.scoring_mode == "llm":
            if HAS_OPENAI:
                api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
                if api_key:
                    self.openai_client = OpenAI(api_key=api_key)
                    logger.info(f"Stage 3 initialized (LLM mode, model: {openai_model})")
                else:
                    logger.error("OPENAI_API_KEY not set")
            else:
                logger.error("OpenAI library not installed")
        elif self.scoring_mode == "hybrid":
            logger.info(f"Stage 3 initialized (Hybrid mode, text: {text_weight}, semantic: {semantic_weight})")
        elif self.scoring_mode == "semantic_only":
            logger.info("Stage 3 initialized (Semantic Only mode)")
    
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
        mode_names = {
            'llm': 'LLM',
            'hybrid': 'Hybrid (Text + Semantic)',
            'semantic_only': 'Semantic Only'
        }
        
        logger.info("=" * 60)
        logger.info(f"Stage 3: {mode_names.get(self.scoring_mode, self.scoring_mode)} Scoring")
        logger.info("=" * 60)
        
        if not stage2_candidates:
            logger.warning("No candidates to score")
            return []
        
        if self.scoring_mode == "llm":
            return self._score_llm(entity_name, stage2_candidates, entity_embedding)
        elif self.scoring_mode == "hybrid":
            return self._score_hybrid(entity_name, stage2_candidates, entity_embedding)
        elif self.scoring_mode == "semantic_only":
            return self._score_semantic(entity_name, stage2_candidates, entity_embedding)
        else:
            logger.error(f"Unknown scoring mode: {self.scoring_mode}")
            return []
    
    def _score_hybrid(
        self,
        entity_name: str,
        candidates: List[Dict[str, Any]],
        entity_embedding: Optional[Any]
    ) -> List[Dict[str, Any]]:
        """Score using hybrid (text + semantic) approach."""
        results = []
        
        for candidate in candidates:
            concept = candidate['concept']
            
            # Text similarity (Jaccard)
            if candidate.get('is_original_standard', True):
                text_sim = self._jaccard_similarity(entity_name, concept.get('concept_name', ''))
            else:
                text_sim = 0.9  # Fixed score for non-std to std
            
            # Semantic similarity
            concept_emb = concept.get('concept_embedding')
            semantic_sim = self._compute_cosine(entity_embedding, concept_emb) or 0.0
            
            # Elasticsearch score (sigmoid normalized)
            es_score = candidate.get('elasticsearch_score', 0.0)
            es_score_normalized = sigmoid_normalize(es_score, center=3.0)
            
            # Combined score
            final_score = self.text_weight * text_sim + self.semantic_weight * semantic_sim
            
            results.append({
                'concept': concept,
                'is_original_standard': candidate.get('is_original_standard', True),
                'original_candidate': candidate.get('original_candidate', {}),
                'elasticsearch_score': es_score,
                'elasticsearch_score_normalized': es_score_normalized,
                'search_type': candidate.get('search_type', 'unknown'),
                'text_similarity': text_sim,
                'semantic_similarity': semantic_sim,
                'final_score': final_score
            })
        
        # Sort by score
        sorted_results = sorted(results, key=lambda x: x['final_score'], reverse=True)
        
        self._log_results("Hybrid", sorted_results)
        return sorted_results
    
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
        if not self.openai_client:
            logger.error("OpenAI client not initialized")
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
            
            if self.include_stage1_scores and entity_embedding is not None:
                concept_emb = concept.get('concept_embedding')
                data['semantic_similarity'] = self._compute_cosine(entity_embedding, concept_emb) or 0.0
            
            results.append(data)
        
        # Get LLM scores
        try:
            llm_scores = self._call_llm(entity_name, results, entity_embedding)
            
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
    
    def _jaccard_similarity(self, text1: str, text2: str, n: int = 3) -> float:
        """Calculate n-gram Jaccard similarity."""
        def ngrams(text: str, n: int) -> set:
            text = text.lower().strip()
            if len(text) < n:
                return {text}
            return {text[i:i+n] for i in range(len(text) - n + 1)}
        
        ng1, ng2 = ngrams(text1, n), ngrams(text2, n)
        if not ng1 or not ng2:
            return 0.0
        
        intersection = len(ng1 & ng2)
        union = len(ng1 | ng2)
        return intersection / union if union > 0 else 0.0
    
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
        candidates: List[Dict[str, Any]],
        entity_embedding: Optional[Any]
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Call LLM API for scoring."""
        prompt = self._build_prompt(entity_name, candidates)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical terminology mapping expert. "
                                   "Select the best OMOP CDM concept for the given entity."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            
            return self._parse_llm_response(response.choices[0].message.content, candidates)
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return None
    
    def _build_prompt(self, entity_name: str, candidates: List[Dict[str, Any]]) -> str:
        """Build LLM prompt."""
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
            if self.include_stage1_scores and 'semantic_similarity' in c:
                info['semantic_similarity'] = round(c['semantic_similarity'], 4)
            candidates_info.append(info)
        
        score_hint = """
**Semantic Similarity Info**:
- semantic_similarity: SapBERT embedding cosine similarity (0.0-1.0)
  - Higher = more semantically similar
  - Use as reference, but prioritize medical accuracy
""" if self.include_stage1_scores else ""
        
        return f"""Select the best OMOP CDM concept for the entity and rank all candidates.

**Entity**: {entity_name}

**Candidates**:
{json.dumps(candidates_info, ensure_ascii=False, indent=2)}
{score_hint}
**Criteria**:
1. Prioritize semantic match between entity and concept name
2. Consider medical context and domain appropriateness
3. IMPORTANT: Map to same or higher level, never to sub-concepts
   - Example: "hypertension" -> "hypertension" or "hypertensive disease", NOT "essential hypertension"
4. Score 0 for completely different concepts

**Output Format** (JSON only):
{{
  "best_match": {{
    "concept_id": "best concept ID",
    "reasoning": "brief reason"
  }},
  "rankings": [
    {{"concept_id": "ID", "rank": 1, "score": 0-5, "reasoning": "reason"}},
    ...
  ]
}}
"""
    
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
            
            if 'best_match' in parsed and 'rankings' in parsed:
                best_id = str(parsed['best_match'].get('concept_id', ''))
                
                for item in parsed['rankings']:
                    cid = str(item.get('concept_id', ''))
                    if cid:
                        result[cid] = {
                            'score': float(item.get('score', 0.0)),
                            'rank': int(item.get('rank', 999)),
                            'reasoning': item.get('reasoning', ''),
                            'is_best_match': cid == best_id
                        }
                
                logger.info(f"LLM selected best match: {best_id}")
            
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
        """Log scoring results."""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Stage 3 {mode} Results:")
        logger.info("=" * 60)
        
        for i, r in enumerate(results[:10], 1):
            concept = r['concept']
            name = concept.get('concept_name', 'N/A')
            cid = concept.get('concept_id', 'N/A')
            score = r.get('final_score', 0.0)
            search_type = r.get('search_type', 'unknown')
            
            logger.info(f"  {i}. {name} (ID: {cid}) [{search_type}]")
            logger.info(f"     Final Score: {score:.4f}")
        
        logger.info("=" * 60)
