"""
PostgreSQL Data Source

Reads OMOP CDM data from PostgreSQL database (internal network CDM).
"""

import logging
from typing import Iterator, Optional

import pandas as pd

from .base import BaseDataSource, DataSourceType


class PostgresDataSource(BaseDataSource):
    """Data source for PostgreSQL database."""
    
    # Default connection settings
    DEFAULT_CONFIG = {
        'host': '172.23.100.146',
        'port': '1341',
        'dbname': 'cdm_public',
        'user': 'cdmreader',
        'password': 'scdm2025!@'
    }
    
    # Default table names
    DEFAULT_TABLES = {
        'concept': 'cdm2024_samples.concept',
        'relationship': 'cdm2024_samples.concept_relationship',
        'synonym': 'cdm2024_samples.concept_synonym'
    }
    
    # 영어 동의어 language_concept_id
    ENGLISH_LANGUAGE_CONCEPT_ID = 4180186
    
    def __init__(
        self,
        host: str = None,
        port: str = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
        concept_table: str = None,
        relationship_table: str = None,
        synonym_table: str = None
    ):
        """
        Initialize PostgreSQL data source.
        
        Args:
            host: Database host (default: 172.23.100.146)
            port: Database port (default: 1341)
            dbname: Database name (default: cdm_public)
            user: Username (default: cdmreader)
            password: Password (default: scdm2025!@)
            concept_table: CONCEPT table name (schema.table)
            relationship_table: CONCEPT_RELATIONSHIP table name
            synonym_table: CONCEPT_SYNONYM table name
        """
        super().__init__(DataSourceType.POSTGRES)
        
        # Connection config
        self.config = {
            'host': host or self.DEFAULT_CONFIG['host'],
            'port': port or self.DEFAULT_CONFIG['port'],
            'dbname': dbname or self.DEFAULT_CONFIG['dbname'],
            'user': user or self.DEFAULT_CONFIG['user'],
            'password': password or self.DEFAULT_CONFIG['password']
        }
        
        # Table names
        self.tables = {
            'concept': concept_table or self.DEFAULT_TABLES['concept'],
            'relationship': relationship_table or self.DEFAULT_TABLES['relationship'],
            'synonym': synonym_table or self.DEFAULT_TABLES['synonym']
        }
        
        # Test connection
        self._test_connection()
        
        self.logger.info(f"Initialized PostgreSQL data source: {self.config['host']}:{self.config['port']}")
    
    def _get_connection(self):
        """Create database connection."""
        try:
            import psycopg2
        except ImportError:
            raise ImportError("psycopg2 required. Install with: pip install psycopg2-binary")
        
        return psycopg2.connect(**self.config)
    
    def _test_connection(self):
        """Test database connection."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            self.logger.info("Database connection successful")
        except Exception as e:
            raise ConnectionError(f"Database connection failed: {e}")
    
    def _get_count(self, table: str) -> int:
        """Get row count for a table."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            return count
        except Exception as e:
            self.logger.error(f"Count query failed for {table}: {e}")
            return 0
    
    def get_concept_count(self) -> int:
        """Return total number of CONCEPT records."""
        return self._get_count(self.tables['concept'])
    
    def get_relationship_count(self) -> int:
        """Return total number of CONCEPT_RELATIONSHIP records."""
        return self._get_count(self.tables['relationship'])
    
    def get_synonym_count(self) -> int:
        """Return total number of CONCEPT_SYNONYM records."""
        return self._get_count(self.tables['synonym'])
    
    def get_concept_small_count(self) -> int:
        """
        Return total number of CONCEPT_SMALL records.
        CONCEPT (Original) + CONCEPT_SYNONYM (영어만, Synonym) 합계
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # CONCEPT 수 + 영어 SYNONYM 수
            query = f"""
                SELECT 
                    (SELECT COUNT(*) FROM {self.tables['concept']}) +
                    (SELECT COUNT(*) FROM {self.tables['synonym']} 
                     WHERE language_concept_id = {self.ENGLISH_LANGUAGE_CONCEPT_ID})
            """
            cursor.execute(query)
            count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            return count
        except Exception as e:
            self.logger.error(f"Count query failed for concept_small: {e}")
            return 0
    
    def _read_table_chunks(
        self,
        table: str,
        columns: list,
        chunk_size: int,
        skip_rows: int,
        max_rows: Optional[int]
    ) -> Iterator[pd.DataFrame]:
        """Read table data in chunks using OFFSET/LIMIT."""
        try:
            conn = self._get_connection()
            
            columns_str = ", ".join(columns)
            base_query = f"SELECT {columns_str} FROM {table} ORDER BY 1"
            
            offset = skip_rows
            total_read = 0
            
            while True:
                if max_rows is not None:
                    remaining = max_rows - total_read
                    if remaining <= 0:
                        break
                    limit = min(chunk_size, remaining)
                else:
                    limit = chunk_size
                
                query = f"{base_query} LIMIT {limit} OFFSET {offset}"
                chunk = pd.read_sql(query, conn)
                
                if len(chunk) == 0:
                    break
                
                chunk.columns = chunk.columns.str.lower()
                total_read += len(chunk)
                offset += len(chunk)
                
                yield chunk
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error reading {table}: {e}")
            raise
    
    def read_concepts(
        self,
        chunk_size: int = 1000,
        skip_rows: int = 0,
        max_rows: Optional[int] = None
    ) -> Iterator[pd.DataFrame]:
        """Read CONCEPT data in chunks."""
        for chunk in self._read_table_chunks(
            self.tables['concept'],
            self.CONCEPT_COLUMNS,
            chunk_size,
            skip_rows,
            max_rows
        ):
            cleaned = self.clean_concept_data(chunk)
            if len(cleaned) > 0:
                yield cleaned
    
    def read_relationships(
        self,
        chunk_size: int = 10000,
        skip_rows: int = 0,
        max_rows: Optional[int] = None
    ) -> Iterator[pd.DataFrame]:
        """Read CONCEPT_RELATIONSHIP data in chunks."""
        for chunk in self._read_table_chunks(
            self.tables['relationship'],
            self.RELATIONSHIP_COLUMNS,
            chunk_size,
            skip_rows,
            max_rows
        ):
            cleaned = self.clean_relationship_data(chunk)
            if len(cleaned) > 0:
                yield cleaned
    
    def read_synonyms(
        self,
        chunk_size: int = 1000,
        skip_rows: int = 0,
        max_rows: Optional[int] = None
    ) -> Iterator[pd.DataFrame]:
        """Read CONCEPT_SYNONYM data in chunks."""
        for chunk in self._read_table_chunks(
            self.tables['synonym'],
            self.SYNONYM_COLUMNS,
            chunk_size,
            skip_rows,
            max_rows
        ):
            cleaned = self.clean_synonym_data(chunk)
            if len(cleaned) > 0:
                yield cleaned
    
    def read_concept_small(
        self,
        chunk_size: int = 1000,
        skip_rows: int = 0,
        max_rows: Optional[int] = None
    ) -> Iterator[pd.DataFrame]:
        """
        Read CONCEPT_SMALL data in chunks.
        
        SQL UNION을 사용하여 CONCEPT + CONCEPT_SYNONYM (영어만)을 합쳐서 반환합니다.
        - CONCEPT: name_type = 'Original'
        - CONCEPT_SYNONYM (language_concept_id=4180186): name_type = 'Synonym'
        """
        try:
            conn = self._get_connection()
            
            concept_table = self.tables['concept']
            synonym_table = self.tables['synonym']
            
            # UNION ALL 쿼리: CONCEPT (Original) + SYNONYM (영어만, JOIN으로 메타데이터 가져옴)
            base_query = f"""
                SELECT 
                    concept_id,
                    concept_name,
                    'Original' AS name_type,
                    domain_id,
                    vocabulary_id,
                    concept_class_id,
                    standard_concept,
                    concept_code,
                    valid_start_date,
                    valid_end_date,
                    invalid_reason
                FROM {concept_table}
                
                UNION ALL
                
                SELECT 
                    c.concept_id,
                    s.concept_synonym_name AS concept_name,
                    'Synonym' AS name_type,
                    c.domain_id,
                    c.vocabulary_id,
                    c.concept_class_id,
                    c.standard_concept,
                    c.concept_code,
                    c.valid_start_date,
                    c.valid_end_date,
                    c.invalid_reason
                FROM {synonym_table} s
                JOIN {concept_table} c ON s.concept_id = c.concept_id
                WHERE s.language_concept_id = {self.ENGLISH_LANGUAGE_CONCEPT_ID}
            """
            
            # 서브쿼리로 감싸서 ORDER BY, LIMIT, OFFSET 적용
            offset = skip_rows
            total_read = 0
            
            while True:
                if max_rows is not None:
                    remaining = max_rows - total_read
                    if remaining <= 0:
                        break
                    limit = min(chunk_size, remaining)
                else:
                    limit = chunk_size
                
                query = f"""
                    SELECT * FROM (
                        {base_query}
                    ) AS concept_small
                    ORDER BY concept_id, name_type
                    LIMIT {limit} OFFSET {offset}
                """
                
                chunk = pd.read_sql(query, conn)
                
                if len(chunk) == 0:
                    break
                
                chunk.columns = chunk.columns.str.lower()
                total_read += len(chunk)
                offset += len(chunk)
                
                # 데이터 정제
                cleaned = self.clean_concept_small_data(chunk)
                if len(cleaned) > 0:
                    yield cleaned
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error reading concept_small: {e}")
            raise
