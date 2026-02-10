"""
Mapping Validation Module

Validates mapping results using LLM to ensure semantic correctness.

Supports multiple LLM providers via LLMClient:
- OpenAI (gpt-4o-mini, etc.)
- SNUH Hari (snuh/hari-q3-14b)
- Google Gemma (google/gemma-3-12b-it)
"""

import json
import logging
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient, get_llm_client

logger = logging.getLogger(__name__)


class MappingValidator:
    """Validates mapping results using LLM."""
    
    # Default LLM hyperparameters (validation uses lower temperature for consistency)
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_TOP_P = 1.0
    
    def __init__(
        self,
        es_client=None,
        llm_client: Optional[LLMClient] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ):
        """
        Initialize validator.
        
        Args:
            es_client: Elasticsearch client for synonym lookup
            llm_client: LLM client instance (uses default if None)
            temperature: LLM temperature (0.0-2.0, default 0.1)
            top_p: LLM top_p / nucleus sampling (0.0-1.0, default 1.0)
        """
        self.es_client = es_client
        self.temperature = temperature if temperature is not None else self.DEFAULT_TEMPERATURE
        self.top_p = top_p if top_p is not None else self.DEFAULT_TOP_P
        
        # LLM client (supports OpenAI, Hari, Gemma)
        self.llm_client = llm_client
        if self.llm_client is None:
            self.llm_client = get_llm_client()
        
        if self.llm_client.is_initialized:
            llm_info = self.llm_client.get_info()
            logger.info(
                f"MappingValidator initialized "
                f"(provider: {llm_info['provider']}, model: {llm_info['model']})"
            )
        else:
            logger.error("LLM client not initialized")
    
    def validate_mapping(
        self,
        entity_name: str,
        concept_id: str,
        concept_name: str,
        synonyms: Optional[List[str]] = None
    ) -> bool:
        """
        Validate a mapping result.
        
        Args:
            entity_name: Input entity name
            concept_id: Mapped concept ID
            concept_name: Mapped concept name
            synonyms: Concept synonyms (fetched from ES if None)
            
        Returns:
            True if mapping is valid
        """
        if not self.llm_client or not self.llm_client.is_initialized:
            logger.error("LLM client not initialized")
            return False
        
        if synonyms is None:
            synonyms = self._fetch_synonyms(concept_id)
        
        try:
            return self._validate_with_llm(entity_name, concept_id, concept_name, synonyms)
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    def _fetch_synonyms(self, concept_id: str) -> List[str]:
        """Fetch synonyms from Elasticsearch."""
        if not self.es_client:
            return []
        
        try:
            synonyms = self.es_client.search_synonyms(str(concept_id))
            logger.debug(f"Fetched {len(synonyms)} synonyms for concept {concept_id}")
            return synonyms
        except Exception as e:
            logger.error(f"Synonym fetch failed: {e}")
            return []
    
    def _validate_with_llm(
        self,
        entity_name: str,
        concept_id: str,
        concept_name: str,
        synonyms: List[str]
    ) -> bool:
        """Validate mapping using LLM."""
        prompt = self._create_prompt(entity_name, concept_id, concept_name, synonyms)
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a medical terminology mapping expert. "
                               "Validate if the input entity matches the mapped OMOP CDM concept."
                },
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_client.chat_completion(
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=256,
                json_mode=True
            )
            
            if response is None:
                return False
            
            result = self._parse_response(response)
            return result
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return False
    
    def _create_prompt(
        self,
        entity_name: str,
        concept_id: str,
        concept_name: str,
        synonyms: List[str]
    ) -> str:
        """Create validation prompt."""
        # Process synonyms (split by semicolons and slashes)
        all_synonyms = []
        for syn in synonyms:
            for part in syn.split(';'):
                for sub in part.split('/'):
                    cleaned = sub.strip()
                    if cleaned and cleaned not in all_synonyms:
                        all_synonyms.append(cleaned)
        
        if all_synonyms:
            synonyms_text = "\n".join([f"- {s}" for s in all_synonyms[:50]])
            prompt = f"""Validate if the following mapping is correct.

**Input Entity**: {entity_name}

**Mapped Concept**:
- Concept ID: {concept_id}
- Concept Name: {concept_name}

**Concept Synonyms** (hints):
{synonyms_text}

**Instructions**:
1. Check if the input entity ({entity_name}) semantically matches the mapped concept ({concept_name}).
2. Use synonyms as hints to determine if the mapping is appropriate.
3. Return True if any synonym matches or is very similar to the input entity.
4. Return False if the input is a sub-concept or completely different concept.

**Output Format** (JSON only):
{{
  "is_valid": true or false,
  "reasoning": "Brief explanation in English"
}}
"""
        else:
            prompt = f"""Validate if the following mapping is correct.

**Input Entity**: {entity_name}

**Mapped Concept**:
- Concept ID: {concept_id}
- Concept Name: {concept_name}

**Instructions**:
1. Check if the input entity ({entity_name}) semantically matches the mapped concept ({concept_name}).
2. Return True if they are semantically equivalent or very similar.
3. Return False if the input is a sub-concept or completely different concept.

**Output Format** (JSON only):
{{
  "is_valid": true or false,
  "reasoning": "Brief explanation in English"
}}
"""
        return prompt
    
    def _parse_response(self, response_text: str) -> bool:
        """Parse LLM response."""
        try:
            text = response_text.strip()
            
            # Remove markdown code blocks
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()
            
            parsed = json.loads(text)
            is_valid = parsed.get('is_valid', False)
            reasoning = parsed.get('reasoning', '')
            
            logger.debug(f"Validation result: {is_valid}, reason: {reasoning}")
            return bool(is_valid)
            
        except Exception as e:
            logger.error(f"Response parsing failed: {e}")
            return False
    
    def validate_candidates_sequentially(
        self,
        entity_name: str,
        candidates: List[Dict[str, Any]],
        max_candidates: int = 10
    ) -> Optional[Dict[str, Any]]:
        """
        Validate candidates sequentially until one passes.
        
        Args:
            entity_name: Input entity name
            candidates: List of candidates (sorted by rank)
            max_candidates: Maximum candidates to validate
            
        Returns:
            First valid candidate or None
        """
        if not candidates:
            logger.warning("No candidates to validate")
            return None
        
        top_candidates = candidates[:max_candidates]
        
        logger.info(f"Sequential validation: {len(top_candidates)} candidates")
        
        for idx, candidate in enumerate(top_candidates, 1):
            concept = candidate.get('concept', {})
            concept_id = str(concept.get('concept_id', ''))
            concept_name = concept.get('concept_name', '')
            
            if not concept_id or not concept_name:
                continue
            
            logger.info(f"  [{idx}/{len(top_candidates)}] Validating: {concept_name}")
            
            is_valid = self.validate_mapping(
                entity_name=entity_name,
                concept_id=concept_id,
                concept_name=concept_name,
                synonyms=None
            )
            
            if is_valid:
                logger.info(f"  Validated: {concept_name}")
                return candidate
            else:
                logger.info(f"  Failed: {concept_name}")
        
        logger.warning(f"All {len(top_candidates)} candidates failed validation")
        return None
