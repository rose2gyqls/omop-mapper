"""
Unified Indexer Module

Main indexer that orchestrates OMOP CDM data indexing from multiple sources.
Supports local CSV and PostgreSQL data sources.

Robust indexing:
    - Checkpoint 파일 기반 재개 (행 번호 기록, 끊긴 위치부터 이어서)
    - Idempotent _id → 재전송 시 덮어쓰기 (중복 없음)
    - 429 지수 백오프 + 개별 실패 문서 재시도 (elasticsearch_indexer에서 처리)
    - 완료 후 데이터 검증 (ES 문서 수 vs 원본 행 수)

Index Names (fixed):
    - concept
    - concept-small
    - concept-relationship
    - concept-synonym
"""

import json
import logging
import os
import time
from typing import Optional, Dict, List

import torch
from tqdm.auto import tqdm

from data_sources import (
    BaseDataSource,
    DataSourceType,
    LocalCSVDataSource,
    PostgresDataSource
)
from sapbert_embedder import SapBERTEmbedder
from elasticsearch_indexer import ElasticsearchIndexer


CHECKPOINT_FILE = '.indexing_checkpoint.json'


class UnifiedIndexer:
    """Unified indexer for OMOP CDM data."""
    
    # Fixed index names
    INDEX_NAMES = {
        'concept': 'concept',
        'concept-small': 'concept-small',
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
        batch_size: int = 512,
        chunk_size: int = 10000,
        include_embeddings: bool = True,
        lowercase: bool = True,
        bulk_delay_sec: float = 0.0
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
            batch_size: Embedding batch size (512+ on GPU recommended)
            chunk_size: Data processing chunk size (10000+ for concept-small recommended)
            include_embeddings: Whether to include SapBERT embeddings
            lowercase: Whether to lowercase concept/synonym names
            bulk_delay_sec: Bulk 요청 간 대기 시간(초, 429 완화용)
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.include_embeddings = include_embeddings
        self.lowercase = lowercase
        self.bulk_delay_sec = bulk_delay_sec
        
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
        self.logger.info(f"Bulk delay: {bulk_delay_sec}s")
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
    
    # =========================================================================
    # Checkpoint 관리
    # =========================================================================
    
    def _load_all_checkpoints(self) -> dict:
        """Load all checkpoints from file."""
        if os.path.exists(CHECKPOINT_FILE):
            try:
                with open(CHECKPOINT_FILE, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _load_checkpoint(self, table: str) -> int:
        """
        Load checkpoint for a table.
        
        Returns:
            rows_completed: 파일 시작부터 성공적으로 인덱싱된 총 행 수 (0이면 처음부터)
        """
        checkpoint = self._load_all_checkpoints()
        entry = checkpoint.get(table, {})
        rows = entry.get('rows_completed', 0)
        if rows > 0:
            self.logger.info(f"Checkpoint loaded: {table} = {rows:,} rows completed")
        return rows
    
    def _save_checkpoint(self, table: str, rows_completed: int):
        """
        Save checkpoint for a table.
        
        Args:
            table: Table name (e.g., 'concept-small')
            rows_completed: 파일 시작부터 성공적으로 인덱싱된 총 행 수
        """
        checkpoint = self._load_all_checkpoints()
        checkpoint[table] = {
            'rows_completed': rows_completed,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        try:
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(checkpoint, f, indent=2)
        except IOError as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")
    
    def _clear_checkpoint(self, table: str):
        """Clear checkpoint for a table (인덱싱 정상 완료 시)."""
        checkpoint = self._load_all_checkpoints()
        checkpoint.pop(table, None)
        if checkpoint:
            try:
                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump(checkpoint, f, indent=2)
            except IOError:
                pass
        else:
            # 모든 테이블 완료 → 파일 삭제
            if os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)
        self.logger.info(f"Checkpoint cleared: {table}")
    
    # =========================================================================
    # Elasticsearch indexer 관리
    # =========================================================================
    
    def _get_indexer(self, index_type: str) -> ElasticsearchIndexer:
        """Get or create Elasticsearch indexer for index type."""
        if index_type not in self._es_indexers:
            index_name = self.INDEX_NAMES[index_type]
            # relationship, synonym은 임베딩 불필요
            include_emb = self.include_embeddings and index_type not in ['relationship', 'synonym']
            
            self._es_indexers[index_type] = ElasticsearchIndexer(
                host=self.es_config['host'],
                port=self.es_config['port'],
                username=self.es_config['username'],
                password=self.es_config['password'],
                index_name=index_name,
                include_embeddings=include_emb
            )
        
        return self._es_indexers[index_type]
    
    def _verify_count(self, indexer: ElasticsearchIndexer, expected_total: int, table: str):
        """인덱싱 완료 후 ES 문서 수 검증."""
        stats = indexer.get_stats()
        es_count = stats.get('document_count', 0) or 0
        
        if es_count >= expected_total:
            self.logger.info(
                f"✓ 검증 완료: {table} - ES {es_count:,} docs (원본 {expected_total:,} rows)"
            )
        else:
            self.logger.warning(
                f"⚠ 수량 불일치: {table} - ES {es_count:,} docs vs 원본 {expected_total:,} rows "
                f"(차이: {expected_total - es_count:,})"
            )
    
    # =========================================================================
    # 테이블별 인덱싱 (Checkpoint 기반)
    # =========================================================================
    
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
            skip_rows: Rows to skip (Checkpoint에서 로드)
            
        Returns:
            True if successful
        """
        table_key = 'concept'
        
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
            
            if actual_max <= 0:
                self.logger.info(f"CONCEPT: 처리할 행 없음 (total={total:,}, skip={skip_rows:,})")
                self._clear_checkpoint(table_key)
                return True
            
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
                    
                    is_last_chunk = (processed + len(chunk)) >= actual_max
                    if indexer.index_documents(
                        docs,
                        show_progress=False,
                        refresh=is_last_chunk,
                        bulk_delay_sec=self.bulk_delay_sec
                    ):
                        # 청크 전체 성공 → Checkpoint 저장
                        indexed += len(docs)
                        processed += len(chunk)
                        pbar.update(len(chunk))
                        self._save_checkpoint(table_key, skip_rows + processed)
                    else:
                        # 청크 실패 → 중단 (재시작 시 이 청크부터 다시)
                        self.logger.error(
                            f"CONCEPT: Chunk failed at offset {skip_rows + processed}. "
                            f"--resume 로 재시작하면 이 위치부터 재개됩니다."
                        )
                        return False
                    
                    if torch.cuda.is_available() and processed % 100000 == 0 and processed > 0:
                        torch.cuda.empty_cache()
            
            elapsed = time.time() - start_time
            
            self.logger.info(f"CONCEPT indexing complete")
            self.logger.info(f"Processed: {processed:,}, Indexed: {indexed:,}")
            self.logger.info(f"Time: {elapsed/60:.1f} min, Speed: {processed/elapsed:.1f} rows/sec")
            
            # 검증 & Checkpoint 정리
            self._verify_count(indexer, total, table_key)
            self._clear_checkpoint(table_key)
            
            return True
            
        except Exception as e:
            self.logger.error(f"CONCEPT indexing failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def index_concept_small(
        self,
        delete_existing: bool = True,
        max_rows: Optional[int] = None,
        skip_rows: int = 0
    ) -> bool:
        """
        Index CONCEPT_SMALL data (merged CONCEPT + SYNONYM).
        """
        table_key = 'concept-small'
        
        self.logger.info("=" * 60)
        self.logger.info("Starting CONCEPT_SMALL indexing")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            indexer = self._get_indexer('concept-small')
            
            if not indexer.create_index(delete_if_exists=delete_existing):
                self.logger.error("Failed to create index")
                return False
            
            total = self.data_source.get_concept_small_count()
            if total == 0:
                self.logger.error(
                    "CONCEPT_SMALL 파일이 없거나 비어있습니다. "
                    "prepare_concept_small.py를 먼저 실행하세요."
                )
                return False
            
            actual_max = min(max_rows or total, total - skip_rows)
            
            if actual_max <= 0:
                self.logger.info(f"CONCEPT_SMALL: 처리할 행 없음 (total={total:,}, skip={skip_rows:,})")
                self._clear_checkpoint(table_key)
                return True
            
            self.logger.info(f"Total records: {total:,}")
            self.logger.info(f"Processing: {actual_max:,} (skip: {skip_rows:,})")
            
            processed = 0
            indexed = 0
            
            with tqdm(total=actual_max, desc="CONCEPT_SMALL", unit="rows") as pbar:
                for chunk in self.data_source.read_concept_small(
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
                    
                    # Generate embeddings (GPU batch)
                    embeddings = None
                    if self.include_embeddings and self.embedder:
                        names = chunk['concept_name'].fillna('').tolist()
                        embeddings = self.embedder.encode(names, show_progress=False)
                    
                    # Convert and index
                    docs = self.data_source.to_es_concept_small(
                        chunk,
                        embeddings=embeddings,
                        include_embeddings=self.include_embeddings
                    )
                    
                    is_last_chunk = (processed + len(chunk)) >= actual_max
                    if indexer.index_documents(
                        docs,
                        show_progress=False,
                        refresh=is_last_chunk,
                        bulk_delay_sec=self.bulk_delay_sec
                    ):
                        indexed += len(docs)
                        processed += len(chunk)
                        pbar.update(len(chunk))
                        self._save_checkpoint(table_key, skip_rows + processed)
                    else:
                        self.logger.error(
                            f"CONCEPT_SMALL: Chunk failed at offset {skip_rows + processed}. "
                            f"--resume 로 재시작하면 이 위치부터 재개됩니다."
                        )
                        return False
                    
                    # Clear GPU cache only periodically to avoid slowdown
                    if torch.cuda.is_available() and processed % 100000 == 0 and processed > 0:
                        torch.cuda.empty_cache()
            
            elapsed = time.time() - start_time
            
            self.logger.info(f"CONCEPT_SMALL indexing complete")
            self.logger.info(f"Processed: {processed:,}, Indexed: {indexed:,}")
            self.logger.info(f"Time: {elapsed/60:.1f} min, Speed: {processed/elapsed:.1f} rows/sec")
            
            self._verify_count(indexer, total, table_key)
            self._clear_checkpoint(table_key)
            
            return True
            
        except Exception as e:
            self.logger.error(f"CONCEPT_SMALL indexing failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def index_relationships(
        self,
        delete_existing: bool = True,
        max_rows: Optional[int] = None,
        skip_rows: int = 0
    ) -> bool:
        """Index CONCEPT_RELATIONSHIP data."""
        table_key = 'relationship'
        
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
            
            if actual_max <= 0:
                self.logger.info(f"RELATIONSHIP: 처리할 행 없음 (total={total:,}, skip={skip_rows:,})")
                self._clear_checkpoint(table_key)
                return True
            
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
                    is_last = (processed + len(chunk)) >= actual_max
                    if indexer.index_documents(
                        docs,
                        show_progress=False,
                        refresh=is_last,
                        bulk_delay_sec=self.bulk_delay_sec
                    ):
                        indexed += len(docs)
                        processed += len(chunk)
                        pbar.update(len(chunk))
                        self._save_checkpoint(table_key, skip_rows + processed)
                    else:
                        self.logger.error(
                            f"RELATIONSHIP: Chunk failed at offset {skip_rows + processed}. "
                            f"--resume 로 재시작하면 이 위치부터 재개됩니다."
                        )
                        return False
            
            elapsed = time.time() - start_time
            
            self.logger.info(f"CONCEPT_RELATIONSHIP indexing complete")
            self.logger.info(f"Processed: {processed:,}, Indexed: {indexed:,}")
            self.logger.info(f"Time: {elapsed/60:.1f} min, Speed: {processed/elapsed:.1f} rows/sec")
            
            self._verify_count(indexer, total, table_key)
            self._clear_checkpoint(table_key)
            
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
        
        단순 조회용 인덱스 (임베딩 없음, 테이블 그대로 인덱싱)
        """
        table_key = 'synonym'
        
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
            
            if actual_max <= 0:
                self.logger.info(f"SYNONYM: 처리할 행 없음 (total={total:,}, skip={skip_rows:,})")
                self._clear_checkpoint(table_key)
                return True
            
            self.logger.info(f"Total records: {total:,}")
            self.logger.info(f"Processing: {actual_max:,} (skip: {skip_rows:,})")
            
            processed = 0
            indexed = 0
            
            # 임베딩 없이 큰 청크로 빠르게 처리
            syn_chunk_size = self.chunk_size * 5
            
            with tqdm(total=actual_max, desc="SYNONYM", unit="rows") as pbar:
                for chunk in self.data_source.read_synonyms(
                    chunk_size=syn_chunk_size,
                    skip_rows=skip_rows,
                    max_rows=actual_max
                ):
                    if len(chunk) == 0:
                        continue
                    
                    docs = self.data_source.to_es_synonyms(
                        chunk,
                        embeddings=None,
                        include_embeddings=False,
                        lowercase=False
                    )
                    is_last = (processed + len(chunk)) >= actual_max
                    if indexer.index_documents(
                        docs,
                        show_progress=False,
                        refresh=is_last,
                        bulk_delay_sec=self.bulk_delay_sec
                    ):
                        indexed += len(docs)
                        processed += len(chunk)
                        pbar.update(len(chunk))
                        self._save_checkpoint(table_key, skip_rows + processed)
                    else:
                        self.logger.error(
                            f"SYNONYM: Chunk failed at offset {skip_rows + processed}. "
                            f"--resume 로 재시작하면 이 위치부터 재개됩니다."
                        )
                        return False
            
            elapsed = time.time() - start_time
            
            self.logger.info(f"CONCEPT_SYNONYM indexing complete")
            self.logger.info(f"Processed: {processed:,}, Indexed: {indexed:,}")
            self.logger.info(f"Time: {elapsed/60:.1f} min, Speed: {processed/elapsed:.1f} rows/sec")
            
            self._verify_count(indexer, total, table_key)
            self._clear_checkpoint(table_key)
            
            return True
            
        except Exception as e:
            self.logger.error(f"CONCEPT_SYNONYM indexing failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    # =========================================================================
    # 전체 인덱싱 오케스트레이션
    # =========================================================================
    
    def index_all(
        self,
        delete_existing: bool = True,
        max_rows: Optional[int] = None,
        tables: List[str] = None
    ) -> Dict[str, bool]:
        """
        Index all tables.
        
        재개 모드(delete_existing=False): Checkpoint 파일에서 각 테이블의
        마지막 성공 위치를 읽어 해당 행부터 재개합니다.
        
        Args:
            delete_existing: Delete existing indices (False = 재개 모드)
            max_rows: Maximum rows per table
            tables: Tables to index (default: concept-small, relationship, synonym)
            
        Returns:
            Dict of table -> success status
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting full indexing")
        if not delete_existing:
            self.logger.info("Mode: RESUME (Checkpoint 기반 재개)")
        else:
            self.logger.info("Mode: FRESH (기존 인덱스 삭제 후 새로 생성)")
        self.logger.info("=" * 60)
        
        if tables is None:
            tables = ['concept-small', 'relationship', 'synonym']
        
        # 재개 모드: Checkpoint에서 skip_rows 로드
        def skip_for(table: str) -> int:
            if delete_existing:
                return 0
            return self._load_checkpoint(table)
        
        results = {}
        
        # concept (원본 concept 테이블만 인덱싱)
        if 'concept' in tables:
            results['concept'] = self.index_concepts(
                delete_existing, max_rows, skip_rows=skip_for('concept'))
        
        # concept-small (concept + synonym merged)
        if 'concept-small' in tables:
            results['concept-small'] = self.index_concept_small(
                delete_existing, max_rows, skip_rows=skip_for('concept-small'))
        
        if 'relationship' in tables:
            results['relationship'] = self.index_relationships(
                delete_existing, max_rows, skip_rows=skip_for('relationship'))
        
        if 'synonym' in tables:
            results['synonym'] = self.index_synonyms(
                delete_existing, max_rows, skip_rows=skip_for('synonym'))
        
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
        source_type: 'local_csv', 'postgres'
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
            concept_small_file=kwargs.get('concept_small_file', 'CONCEPT_SMALL.csv'),
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
    
    else:
        raise ValueError(f"Unknown source type: {source_type}")
