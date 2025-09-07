"""
OMOP CONCEPT Elasticsearch 인덱싱 모듈

이 패키지는 OMOP CDM CONCEPT 데이터를 SapBERT 임베딩과 함께 
Elasticsearch에 인덱싱하는 기능을 제공합니다.

두 가지 방식을 지원합니다:
1. 로컬 임베딩 생성 (sapbert_embedder 사용)
2. Eland + Ingest Pipeline (eland_model_manager 사용) - 권장
"""

# 로컬 임베딩 생성 방식
from .sapbert_embedder import SapBERTEmbedder
from .main_indexer import ConceptIndexingPipeline

# Eland + Ingest Pipeline 방식 (권장)
from .eland_model_manager import ElandModelManager  
from .main_indexer_eland import ConceptIndexingPipelineEland

# 공통 모듈
from .elasticsearch_indexer import ConceptElasticsearchIndexer
from .concept_data_processor import ConceptDataProcessor

__version__ = "2.0.0"
__author__ = "rose"

__all__ = [
    # 로컬 임베딩 방식
    "SapBERTEmbedder",
    "ConceptIndexingPipeline",
    
    # Eland 방식 (권장)
    "ElandModelManager",
    "ConceptIndexingPipelineEland",
    
    # 공통
    "ConceptElasticsearchIndexer", 
    "ConceptDataProcessor"
]
