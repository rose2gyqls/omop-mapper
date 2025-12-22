"""
Unified Indexer Module

Main indexer that orchestrates OMOP CDM data indexing from multiple sources.
Supports local CSV, PostgreSQL, and Athena API data sources.

Index Names (fixed):
    - concept
    - concept-relationship
    - concept-synonym
"""

import logging
import time
from typing import Optional, Dict, List

import torch
from tqdm.auto import tqdm

from data_sources import (
    BaseDataSource,
    DataSourceType,
    LocalCSVDataSource,
    PostgresDataSource,
    AthenaAPIDataSource
)
from sapbert_embedder import SapBERTEmbedder
from elasticsearch_indexer import ElasticsearchIndexer


class UnifiedIndexer:
    """Unified indexer for OMOP CDM data."""
    
    # Fixed index names
    INDEX_NAMES = {
        'concept': 'concept',
        'relationship': 'concept-relationship',
        'synonym': 'concept-synonym'
    }
    
    def __init__(
        self,
        data_source: BaseDataSource,
        es_host: str = "3.35.110.161",
        es_port: int = 9200,
        es_username: str = "elastic",
        es_password: str = "snomed",
        model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        gpu_device: int = 0,
        batch_size: int = 128,
        chunk_size: int = 1000,
        include_embeddings: bool = True,
        lowercase: bool = True
    ):
        """
        Initialize unified indexer.
        
        Args:
            data_source: Data source instance
            es_host: Elasticsearch host
            es_port: Elasticsearch port
            es_username: Elasticsearch username
            es_password: Elasticsearch password
            model_name: SapBERT model name
            gpu_device: GPU device number (-1 for CPU)
            batch_size: Embedding batch size
            chunk_size: Data processing chunk size
            include_embeddings: Whether to include SapBERT embeddings
            lowercase: Whether to lowercase concept/synonym names
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.include_embeddings = include_embeddings
        self.lowercase = lowercase
        
        self.es_config = {
            'host': es_host,
            'port': es_port,
            'username': es_username,
            'password': es_password
        }
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Device setup
        if gpu_device >= 0 and torch.cuda.is_available():
            self.device = f"cuda:{gpu_device}"
        else:
            self.device = "cpu"
        
        self.logger.info("=" * 60)
        self.logger.info("Initializing Unified Indexer")
        self.logger.info(f"Data source: {data_source.source_type.value}")
        self.logger.info(f"Elasticsearch: {es_host}:{es_port}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Include embeddings: {include_embeddings}")
        self.logger.info(f"Lowercase: {lowercase}")
        self.logger.info("=" * 60)
        
        # Initialize SapBERT embedder
        self.embedder = None
        if include_embeddings:
            self.logger.info("Loading SapBERT model...")
            self.embedder = SapBERTEmbedder(
                model_name=model_name,
                device=self.device,
                batch_size=batch_size
            )
        
        # Elasticsearch indexers (lazy initialization)
        self._es_indexers = {}
    
    def _get_indexer(self, index_type: str) -> ElasticsearchIndexer:
        """Get or create Elasticsearch indexer for index type."""
        if index_type not in self._es_indexers:
            index_name = self.INDEX_NAMES[index_type]
            include_emb = self.include_embeddings and index_type != 'relationship'
            
            self._es_indexers[index_type] = ElasticsearchIndexer(
                host=self.es_config['host'],
                port=self.es_config['port'],
                username=self.es_config['username'],
                password=self.es_config['password'],
                index_name=index_name,
                include_embeddings=include_emb
            )
        
        return self._es_indexers[index_type]
    
    def index_concepts(
        self,
        delete_existing: bool = True,
        max_rows: Optional[int] = None,
        skip_rows: int = 0
    ) -> bool:
        """
        Index CONCEPT data.
        
        Args:
            delete_existing: Delete existing index
            max_rows: Maximum rows to process
            skip_rows: Rows to skip
            
        Returns:
            True if successful
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting CONCEPT indexing")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            indexer = self._get_indexer('concept')
            
            if not indexer.create_index(delete_if_exists=delete_existing):
                self.logger.error("Failed to create index")
                return False
            
            total = self.data_source.get_concept_count()
            actual_max = min(max_rows or total, total - skip_rows)
            
            self.logger.info(f"Total records: {total:,}")
            self.logger.info(f"Processing: {actual_max:,} (skip: {skip_rows:,})")
            
            processed = 0
            indexed = 0
            
            with tqdm(total=actual_max, desc="CONCEPT", unit="rows") as pbar:
                for chunk in self.data_source.read_concepts(
                    chunk_size=self.chunk_size,
                    skip_rows=skip_rows,
                    max_rows=actual_max
                ):
                    if len(chunk) == 0:
                        continue
                    
                    # Lowercase if enabled
                    if self.lowercase:
                        chunk = chunk.copy()
                        chunk['concept_name'] = chunk['concept_name'].str.lower()
                    
                    # Generate embeddings
                    embeddings = None
                    if self.include_embeddings and self.embedder:
                        names = chunk['concept_name'].fillna('').tolist()
                        embeddings = self.embedder.encode(names, show_progress=False)
                    
                    # Convert and index
                    docs = self.data_source.to_es_concepts(
                        chunk,
                        embeddings=embeddings,
                        include_embeddings=self.include_embeddings
                    )
                    
                    if indexer.index_documents(docs, show_progress=False):
                        indexed += len(docs)
                    
                    processed += len(chunk)
                    pbar.update(len(chunk))
                    
                    # Clear GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            elapsed = time.time() - start_time
            stats = indexer.get_stats()
            
            self.logger.info(f"CONCEPT indexing complete")
            self.logger.info(f"Processed: {processed:,}, Indexed: {indexed:,}")
            self.logger.info(f"Time: {elapsed/60:.1f} min, Speed: {processed/elapsed:.1f} rows/sec")
            self.logger.info(f"Index stats: {stats}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"CONCEPT indexing failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def index_relationships(
        self,
        delete_existing: bool = True,
        max_rows: Optional[int] = None,
        skip_rows: int = 0
    ) -> bool:
        """
        Index CONCEPT_RELATIONSHIP data.
        
        Args:
            delete_existing: Delete existing index
            max_rows: Maximum rows to process
            skip_rows: Rows to skip
            
        Returns:
            True if successful
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting CONCEPT_RELATIONSHIP indexing")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            indexer = self._get_indexer('relationship')
            
            if not indexer.create_index(delete_if_exists=delete_existing):
                self.logger.error("Failed to create index")
                return False
            
            total = self.data_source.get_relationship_count()
            actual_max = min(max_rows or total, total - skip_rows)
            
            self.logger.info(f"Total records: {total:,}")
            self.logger.info(f"Processing: {actual_max:,} (skip: {skip_rows:,})")
            
            processed = 0
            indexed = 0
            
            # Use larger chunks for relationships (no embeddings)
            rel_chunk_size = self.chunk_size * 10
            
            with tqdm(total=actual_max, desc="RELATIONSHIP", unit="rows") as pbar:
                for chunk in self.data_source.read_relationships(
                    chunk_size=rel_chunk_size,
                    skip_rows=skip_rows,
                    max_rows=actual_max
                ):
                    if len(chunk) == 0:
                        continue
                    
                    docs = self.data_source.to_es_relationships(chunk)
                    
                    if indexer.index_documents(docs, show_progress=False):
                        indexed += len(docs)
                    
                    processed += len(chunk)
                    pbar.update(len(chunk))
            
            elapsed = time.time() - start_time
            stats = indexer.get_stats()
            
            self.logger.info(f"CONCEPT_RELATIONSHIP indexing complete")
            self.logger.info(f"Processed: {processed:,}, Indexed: {indexed:,}")
            self.logger.info(f"Time: {elapsed/60:.1f} min, Speed: {processed/elapsed:.1f} rows/sec")
            self.logger.info(f"Index stats: {stats}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"CONCEPT_RELATIONSHIP indexing failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def index_synonyms(
        self,
        delete_existing: bool = True,
        max_rows: Optional[int] = None,
        skip_rows: int = 0
    ) -> bool:
        """
        Index CONCEPT_SYNONYM data.
        
        Args:
            delete_existing: Delete existing index
            max_rows: Maximum rows to process
            skip_rows: Rows to skip
            
        Returns:
            True if successful
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting CONCEPT_SYNONYM indexing")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            indexer = self._get_indexer('synonym')
            
            if not indexer.create_index(delete_if_exists=delete_existing):
                self.logger.error("Failed to create index")
                return False
            
            total = self.data_source.get_synonym_count()
            actual_max = min(max_rows or total, total - skip_rows)
            
            self.logger.info(f"Total records: {total:,}")
            self.logger.info(f"Processing: {actual_max:,} (skip: {skip_rows:,})")
            
            processed = 0
            indexed = 0
            
            with tqdm(total=actual_max, desc="SYNONYM", unit="rows") as pbar:
                for chunk in self.data_source.read_synonyms(
                    chunk_size=self.chunk_size,
                    skip_rows=skip_rows,
                    max_rows=actual_max
                ):
                    if len(chunk) == 0:
                        continue
                    
                    # Get names for embedding
                    names = chunk['concept_synonym_name'].fillna('').tolist()
                    if self.lowercase:
                        names = [n.lower() if n else "" for n in names]
                    
                    # Generate embeddings
                    embeddings = None
                    if self.include_embeddings and self.embedder:
                        embeddings = self.embedder.encode(names, show_progress=False)
                    
                    # Convert and index
                    docs = self.data_source.to_es_synonyms(
                        chunk,
                        embeddings=embeddings,
                        include_embeddings=self.include_embeddings,
                        lowercase=self.lowercase
                    )
                    
                    if indexer.index_documents(docs, show_progress=False):
                        indexed += len(docs)
                    
                    processed += len(chunk)
                    pbar.update(len(chunk))
                    
                    # Clear GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            elapsed = time.time() - start_time
            stats = indexer.get_stats()
            
            self.logger.info(f"CONCEPT_SYNONYM indexing complete")
            self.logger.info(f"Processed: {processed:,}, Indexed: {indexed:,}")
            self.logger.info(f"Time: {elapsed/60:.1f} min, Speed: {processed/elapsed:.1f} rows/sec")
            self.logger.info(f"Index stats: {stats}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"CONCEPT_SYNONYM indexing failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def index_all(
        self,
        delete_existing: bool = True,
        max_rows: Optional[int] = None,
        tables: List[str] = None
    ) -> Dict[str, bool]:
        """
        Index all tables.
        
        Args:
            delete_existing: Delete existing indices
            max_rows: Maximum rows per table
            tables: Tables to index (default: all)
            
        Returns:
            Dict of table -> success status
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting full indexing")
        self.logger.info("=" * 60)
        
        if tables is None:
            tables = ['concept', 'relationship', 'synonym']
        
        results = {}
        
        if 'concept' in tables:
            results['concept'] = self.index_concepts(delete_existing, max_rows)
        
        if 'relationship' in tables:
            results['relationship'] = self.index_relationships(delete_existing, max_rows)
        
        if 'synonym' in tables:
            results['synonym'] = self.index_synonyms(delete_existing, max_rows)
        
        self.logger.info("=" * 60)
        self.logger.info(f"Indexing complete: {results}")
        self.logger.info("=" * 60)
        
        return results
    
    def cleanup(self):
        """Release resources."""
        self.logger.info("Cleaning up resources...")
        
        if self.embedder:
            self.embedder.cleanup()
            self.embedder = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_data_source(source_type: str, **kwargs) -> BaseDataSource:
    """
    Factory function to create data source.
    
    Args:
        source_type: 'local_csv', 'postgres', or 'athena_api'
        **kwargs: Data source specific arguments
        
    Returns:
        BaseDataSource instance
    """
    if source_type == 'local_csv':
        return LocalCSVDataSource(
            data_folder=kwargs.get('data_folder'),
            concept_file=kwargs.get('concept_file', 'CONCEPT.csv'),
            relationship_file=kwargs.get('relationship_file', 'CONCEPT_RELATIONSHIP.csv'),
            synonym_file=kwargs.get('synonym_file', 'CONCEPT_SYNONYM.csv'),
            delimiter=kwargs.get('delimiter', '\t')
        )
    
    elif source_type == 'postgres':
        return PostgresDataSource(
            host=kwargs.get('host'),
            port=kwargs.get('port'),
            dbname=kwargs.get('dbname'),
            user=kwargs.get('user'),
            password=kwargs.get('password'),
            concept_table=kwargs.get('concept_table'),
            relationship_table=kwargs.get('relationship_table'),
            synonym_table=kwargs.get('synonym_table')
        )
    
    elif source_type == 'athena_api':
        return AthenaAPIDataSource(
            vocabularies=kwargs.get('vocabularies'),
            domains=kwargs.get('domains'),
            standard_only=kwargs.get('standard_only', True),
            page_size=kwargs.get('page_size', 100),
            rate_limit_delay=kwargs.get('rate_limit_delay', 0.5)
        )
    
    else:
        raise ValueError(f"Unknown source type: {source_type}")
