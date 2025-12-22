"""
OMOP CDM Indexing Module

This module provides tools for indexing OMOP CDM data into Elasticsearch.
Supports three data sources: local CSV, PostgreSQL, and Athena API.

Components:
    - data_sources: Data source adapters for different CDM sources
    - elasticsearch_indexer: Elasticsearch indexing utilities
    - sapbert_embedder: SapBERT embedding generator
    - unified_indexer: Main indexer that orchestrates the indexing process
"""

from .elasticsearch_indexer import ElasticsearchIndexer
from .sapbert_embedder import SapBERTEmbedder
from .unified_indexer import UnifiedIndexer, create_data_source

__all__ = [
    'ElasticsearchIndexer',
    'SapBERTEmbedder',
    'UnifiedIndexer',
    'create_data_source'
]
