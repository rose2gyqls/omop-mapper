"""
MapOMOP

A 3-stage medical entity mapping system for OMOP CDM.

Stages:
    1. Candidate Retrieval: Multi-strategy search (lexical, semantic, combined)
    2. Standard Collection: Convert non-standard to standard concepts
    3. Hybrid Scoring: LLM or embedding-based final ranking

LLM Providers:
    - OpenAI (gpt-4o-mini, etc.)
    - SNUH Hari (snuh/hari-q3-14b via vLLM)
    - Google Gemma (google/gemma-3-12b-it via vLLM)

Usage:
    from MapOMOP import EntityMappingAPI, EntityInput, DomainID
    
    api = EntityMappingAPI()
    entity = EntityInput(entity_name="aspirin", domain_id=DomainID.DRUG)
    results = api.map_entity(entity)
"""

from .elasticsearch_client import ElasticsearchClient
from .entity_mapping_api import (
    DomainID,
    EntityInput,
    EntityMappingAPI,
    MappingResult,
)
from .llm_client import (
    LLMClient,
    LLMProvider,
    get_llm_client,
    create_llm_client,
)
from .mapping_stages import ScoringMode
from .utils import sigmoid_normalize

__version__ = "1.0.0"
__author__ = "rose"

__all__ = [
    # Main API
    "EntityMappingAPI",
    "EntityInput",
    "DomainID",
    "MappingResult",
    # Scoring Mode
    "ScoringMode",
    # LLM Client
    "LLMClient",
    "LLMProvider",
    "get_llm_client",
    "create_llm_client",
    # Elasticsearch
    "ElasticsearchClient",
    # Utils
    "sigmoid_normalize",
]
