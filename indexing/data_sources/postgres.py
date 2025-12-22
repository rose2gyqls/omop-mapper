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
