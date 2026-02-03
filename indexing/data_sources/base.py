"""
Base Data Source Module

Defines the abstract base class that all data sources must implement.
Provides common data cleaning and conversion utilities.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Iterator, Optional

import pandas as pd
import numpy as np


class DataSourceType(Enum):
    """Enumeration of supported data source types."""
    LOCAL_CSV = "local_csv"
    POSTGRES = "postgres"
    ATHENA_API = "athena_api"


class BaseDataSource(ABC):
    """Abstract base class for all data sources."""
    
    # CONCEPT table columns
    CONCEPT_COLUMNS = [
        'concept_id', 'concept_name', 'domain_id', 'vocabulary_id',
        'concept_class_id', 'standard_concept', 'concept_code',
        'valid_start_date', 'valid_end_date', 'invalid_reason'
    ]
    
    # CONCEPT_RELATIONSHIP table columns
    RELATIONSHIP_COLUMNS = [
        'concept_id_1', 'concept_id_2', 'relationship_id',
        'valid_start_date', 'valid_end_date', 'invalid_reason'
    ]
    
    # CONCEPT_SYNONYM table columns
    SYNONYM_COLUMNS = [
        'concept_id', 'concept_synonym_name', 'language_concept_id'
    ]
    
    # CONCEPT_SMALL table columns (CONCEPT + SYNONYM merged)
    CONCEPT_SMALL_COLUMNS = [
        'concept_id', 'concept_name', 'name_type', 'domain_id', 'vocabulary_id',
        'concept_class_id', 'standard_concept', 'concept_code',
        'valid_start_date', 'valid_end_date', 'invalid_reason'
    ]
    
    def __init__(self, source_type: DataSourceType):
        """
        Initialize data source.
        
        Args:
            source_type: Type of data source
        """
        self.source_type = source_type
        self.logger = logging.getLogger(self.__class__.__name__)
    
    # Abstract methods that subclasses must implement
    
    @abstractmethod
    def get_concept_count(self) -> int:
        """Return total number of CONCEPT records."""
        pass
    
    @abstractmethod
    def get_relationship_count(self) -> int:
        """Return total number of CONCEPT_RELATIONSHIP records."""
        pass
    
    @abstractmethod
    def get_synonym_count(self) -> int:
        """Return total number of CONCEPT_SYNONYM records."""
        pass
    
    @abstractmethod
    def read_concepts(
        self,
        chunk_size: int = 1000,
        skip_rows: int = 0,
        max_rows: Optional[int] = None
    ) -> Iterator[pd.DataFrame]:
        """Read CONCEPT data in chunks."""
        pass
    
    @abstractmethod
    def read_relationships(
        self,
        chunk_size: int = 10000,
        skip_rows: int = 0,
        max_rows: Optional[int] = None
    ) -> Iterator[pd.DataFrame]:
        """Read CONCEPT_RELATIONSHIP data in chunks."""
        pass
    
    @abstractmethod
    def read_synonyms(
        self,
        chunk_size: int = 1000,
        skip_rows: int = 0,
        max_rows: Optional[int] = None
    ) -> Iterator[pd.DataFrame]:
        """Read CONCEPT_SYNONYM data in chunks."""
        pass
    
    def get_concept_small_count(self) -> int:
        """Return total number of CONCEPT_SMALL records."""
        # 기본 구현: 서브클래스에서 오버라이드 가능
        return 0
    
    def read_concept_small(
        self,
        chunk_size: int = 1000,
        skip_rows: int = 0,
        max_rows: Optional[int] = None
    ) -> Iterator[pd.DataFrame]:
        """Read CONCEPT_SMALL data in chunks."""
        # 기본 구현: 서브클래스에서 오버라이드
        return iter([])
    
    # Common utility methods
    
    def validate_columns(
        self,
        df: pd.DataFrame,
        expected_columns: List[str],
        table_name: str
    ) -> pd.DataFrame:
        """
        Validate and select expected columns from DataFrame.
        
        Args:
            df: Input DataFrame
            expected_columns: List of expected column names
            table_name: Table name for logging
            
        Returns:
            DataFrame with only expected columns
        """
        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Find available columns
        available = [col for col in expected_columns if col in df.columns]
        missing = set(expected_columns) - set(df.columns)
        
        if missing:
            self.logger.warning(f"{table_name}: Missing columns: {missing}")
        
        return df[available].copy()
    
    def clean_concept_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean CONCEPT data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Remove rows with null concept_id or concept_name
        df = df.dropna(subset=['concept_id', 'concept_name'])
        df['concept_id'] = df['concept_id'].astype(str)
        
        # Clean text columns
        text_cols = ['concept_name', 'domain_id', 'vocabulary_id', 'concept_class_id']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['', 'None', 'nan', 'NaN'], None)
        
        # Validate date columns
        for col in ['valid_start_date', 'valid_end_date']:
            if col in df.columns:
                df[col] = df[col].apply(self._parse_date)
        
        # Validate standard_concept (only 'S' or 'C' allowed)
        if 'standard_concept' in df.columns:
            df['standard_concept'] = df['standard_concept'].apply(
                lambda x: x if x in ['S', 'C'] else None
            )
        
        return df
    
    def clean_relationship_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean CONCEPT_RELATIONSHIP data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Remove rows with null IDs
        df = df.dropna(subset=['concept_id_1', 'concept_id_2'])
        df['concept_id_1'] = df['concept_id_1'].astype(str)
        df['concept_id_2'] = df['concept_id_2'].astype(str)
        
        if 'relationship_id' in df.columns:
            df['relationship_id'] = df['relationship_id'].astype(str).str.strip()
        
        # Validate date columns
        for col in ['valid_start_date', 'valid_end_date']:
            if col in df.columns:
                df[col] = df[col].apply(self._parse_date)
        
        return df
    
    def clean_synonym_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean CONCEPT_SYNONYM data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Remove rows with null values
        df = df.dropna(subset=['concept_id', 'concept_synonym_name'])
        df['concept_id'] = df['concept_id'].astype(str)
        
        if 'concept_synonym_name' in df.columns:
            df['concept_synonym_name'] = df['concept_synonym_name'].astype(str).str.strip()
            df['concept_synonym_name'] = df['concept_synonym_name'].replace(['', 'None', 'nan'], None)
            df = df.dropna(subset=['concept_synonym_name'])
        
        if 'language_concept_id' in df.columns:
            df['language_concept_id'] = df['language_concept_id'].astype(str).str.strip()
        
        return df
    
    def clean_concept_small_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean CONCEPT_SMALL data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Remove rows with null concept_id or concept_name
        df = df.dropna(subset=['concept_id', 'concept_name'])
        df['concept_id'] = df['concept_id'].astype(str)
        
        # Clean text columns
        text_cols = ['concept_name', 'name_type', 'domain_id', 'vocabulary_id', 'concept_class_id']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['', 'None', 'nan', 'NaN'], None)
        
        # Validate date columns
        for col in ['valid_start_date', 'valid_end_date']:
            if col in df.columns:
                df[col] = df[col].apply(self._parse_date)
        
        # Validate standard_concept (only 'S' or 'C' allowed)
        if 'standard_concept' in df.columns:
            df['standard_concept'] = df['standard_concept'].apply(
                lambda x: x if x in ['S', 'C'] else None
            )
        
        return df
    
    def _parse_date(self, date_value) -> Optional[str]:
        """
        Parse date value to YYYYMMDD format.
        
        Args:
            date_value: Date value (various formats)
            
        Returns:
            Date string in YYYYMMDD format or None
        """
        if pd.isna(date_value) or date_value is None:
            return None
        
        date_str = str(date_value).strip()
        
        if date_str in ['', 'None', 'nan', 'NaT']:
            return None
        
        # Already YYYYMMDD format
        if len(date_str) == 8 and date_str.isdigit():
            year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
            if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                return date_str
        
        # Try other formats
        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y']:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y%m%d')
            except ValueError:
                continue
        
        return None
    
    # Conversion methods for Elasticsearch
    
    def to_es_concepts(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray = None,
        include_embeddings: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Convert CONCEPT DataFrame to Elasticsearch documents.
        
        Args:
            df: CONCEPT DataFrame
            embeddings: Optional embedding vectors
            include_embeddings: Whether to include embeddings
            
        Returns:
            List of Elasticsearch documents
        """
        documents = []
        
        for i, (_, row) in enumerate(df.iterrows()):
            doc = {col: self._to_str(row.get(col)) for col in self.CONCEPT_COLUMNS if col in df.columns}
            
            if include_embeddings and embeddings is not None and i < len(embeddings):
                doc['concept_embedding'] = embeddings[i].tolist()
            
            documents.append(doc)
        
        return documents
    
    def to_es_relationships(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert CONCEPT_RELATIONSHIP DataFrame to Elasticsearch documents.
        
        Args:
            df: CONCEPT_RELATIONSHIP DataFrame
            
        Returns:
            List of Elasticsearch documents
        """
        documents = []
        
        for _, row in df.iterrows():
            doc = {col: self._to_str(row.get(col)) for col in self.RELATIONSHIP_COLUMNS if col in df.columns}
            documents.append(doc)
        
        return documents
    
    def to_es_synonyms(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray = None,
        include_embeddings: bool = True,
        lowercase: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Convert CONCEPT_SYNONYM DataFrame to Elasticsearch documents.
        
        Args:
            df: CONCEPT_SYNONYM DataFrame
            embeddings: Optional embedding vectors
            include_embeddings: Whether to include embeddings
            lowercase: Whether to lowercase synonym names
            
        Returns:
            List of Elasticsearch documents
        """
        documents = []
        
        for i, (_, row) in enumerate(df.iterrows()):
            doc = {col: self._to_str(row.get(col)) for col in self.SYNONYM_COLUMNS if col in df.columns}
            
            if lowercase and doc.get('concept_synonym_name'):
                doc['concept_synonym_name'] = doc['concept_synonym_name'].lower()
            
            if include_embeddings and embeddings is not None and i < len(embeddings):
                doc['concept_synonym_embedding'] = embeddings[i].tolist()
            
            documents.append(doc)
        
        return documents
    
    def to_es_concept_small(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray = None,
        include_embeddings: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Convert CONCEPT_SMALL DataFrame to Elasticsearch documents.
        
        Args:
            df: CONCEPT_SMALL DataFrame
            embeddings: Optional embedding vectors
            include_embeddings: Whether to include embeddings
            
        Returns:
            List of Elasticsearch documents
        """
        documents = []
        
        for i, (_, row) in enumerate(df.iterrows()):
            doc = {col: self._to_str(row.get(col)) for col in self.CONCEPT_SMALL_COLUMNS if col in df.columns}
            
            if include_embeddings and embeddings is not None and i < len(embeddings):
                doc['concept_embedding'] = embeddings[i].tolist()
            
            documents.append(doc)
        
        return documents
    
    def _to_str(self, value) -> Optional[str]:
        """Convert value to string, handling None/NaN."""
        if pd.isna(value) or value is None:
            return None
        return str(value)
