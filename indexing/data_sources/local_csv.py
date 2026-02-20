"""
Local CSV Data Source

Reads OMOP CDM data from local CSV files.
Expects tab-separated files: CONCEPT.csv, CONCEPT_RELATIONSHIP.csv, CONCEPT_SYNONYM.csv
"""

import logging
import subprocess
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from .base import BaseDataSource, DataSourceType


class LocalCSVDataSource(BaseDataSource):
    """Data source for local OMOP CDM CSV files."""
    
    def __init__(
        self,
        data_folder: str,
        concept_file: str = "CONCEPT.csv",
        relationship_file: str = "CONCEPT_RELATIONSHIP.csv",
        synonym_file: str = "CONCEPT_SYNONYM.csv",
        concept_small_file: str = "CONCEPT_SMALL.csv",
        delimiter: str = "\t",
        relationship_delimiter: Optional[str] = ","
    ):
        """
        Initialize local CSV data source.
        
        Args:
            data_folder: Path to folder containing CSV files
            concept_file: CONCEPT file name
            relationship_file: CONCEPT_RELATIONSHIP file name
            synonym_file: CONCEPT_SYNONYM file name
            concept_small_file: CONCEPT_SMALL file name (merged concept + synonym)
            delimiter: CSV delimiter (default: tab)
            relationship_delimiter: Delimiter for CONCEPT_RELATIONSHIP file (default: comma).
                Use '\t' if your relationship file is tab-separated.
        """
        super().__init__(DataSourceType.LOCAL_CSV)
        
        self.data_folder = Path(data_folder)
        self.delimiter = delimiter
        self.relationship_delimiter = relationship_delimiter if relationship_delimiter is not None else delimiter
        
        # File paths
        self.concept_path = self.data_folder / concept_file
        self.relationship_path = self.data_folder / relationship_file
        self.synonym_path = self.data_folder / synonym_file
        self.concept_small_path = self.data_folder / concept_small_file
        
        # Validate
        self._validate()
        
        self.logger.info(f"Initialized local CSV data source: {data_folder}")
    
    def _validate(self):
        """Validate data folder and files exist."""
        if not self.data_folder.exists():
            raise FileNotFoundError(f"Data folder not found: {self.data_folder}")
        
        for name, path in [
            ("CONCEPT", self.concept_path),
            ("CONCEPT_RELATIONSHIP", self.relationship_path),
            ("CONCEPT_SYNONYM", self.synonym_path),
            ("CONCEPT_SMALL", self.concept_small_path)
        ]:
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                self.logger.info(f"{name}: {path} ({size_mb:.1f} MB)")
            else:
                self.logger.warning(f"{name}: File not found at {path}")
    
    def _count_lines(self, file_path: Path) -> int:
        """Count lines in file (excluding header)."""
        if not file_path.exists():
            return 0
        
        try:
            result = subprocess.run(
                ["wc", "-l", str(file_path)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return max(0, int(result.stdout.split()[0]) - 1)
        except Exception:
            pass
        
        # Fallback: count manually
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f) - 1
    
    def get_concept_count(self) -> int:
        """Return total number of CONCEPT records."""
        return self._count_lines(self.concept_path)
    
    def get_relationship_count(self) -> int:
        """Return total number of CONCEPT_RELATIONSHIP records."""
        return self._count_lines(self.relationship_path)
    
    def get_synonym_count(self) -> int:
        """Return total number of CONCEPT_SYNONYM records."""
        return self._count_lines(self.synonym_path)
    
    def get_concept_small_count(self) -> int:
        """Return total number of CONCEPT_SMALL records."""
        return self._count_lines(self.concept_small_path)
    
    def _read_csv_chunks(
        self,
        file_path: Path,
        expected_columns: list,
        table_name: str,
        chunk_size: int,
        skip_rows: int,
        max_rows: Optional[int],
        delimiter: Optional[str] = None
    ) -> Iterator[pd.DataFrame]:
        """Generic CSV chunk reader."""
        if not file_path.exists():
            self.logger.error(f"{table_name} file not found: {file_path}")
            return

        sep = delimiter if delimiter is not None else self.delimiter
        try:
            params = {
                'sep': sep,
                'chunksize': chunk_size,
                'skiprows': list(range(1, skip_rows + 1)) if skip_rows > 0 else None,
                'nrows': max_rows,
                'low_memory': False,
                'dtype': str,
                'na_values': ['', 'NULL', 'null', 'None', 'NA'],
                'keep_default_na': True
            }
            
            first_chunk = True
            
            for chunk in pd.read_csv(file_path, **params):
                if first_chunk:
                    chunk = self.validate_columns(chunk, expected_columns, table_name)
                    first_chunk = False
                
                if len(chunk) > 0:
                    yield chunk
                    
        except Exception as e:
            self.logger.error(f"Error reading {table_name}: {e}")
            raise
    
    def read_concepts(
        self,
        chunk_size: int = 1000,
        skip_rows: int = 0,
        max_rows: Optional[int] = None
    ) -> Iterator[pd.DataFrame]:
        """Read CONCEPT data in chunks."""
        for chunk in self._read_csv_chunks(
            self.concept_path,
            self.CONCEPT_COLUMNS,
            "CONCEPT",
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
        for chunk in self._read_csv_chunks(
            self.relationship_path,
            self.RELATIONSHIP_COLUMNS,
            "CONCEPT_RELATIONSHIP",
            chunk_size,
            skip_rows,
            max_rows,
            delimiter=self.relationship_delimiter
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
        for chunk in self._read_csv_chunks(
            self.synonym_path,
            self.SYNONYM_COLUMNS,
            "CONCEPT_SYNONYM",
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
        """Read CONCEPT_SMALL data in chunks."""
        if not self.concept_small_path.exists():
            self.logger.error(
                f"CONCEPT_SMALL 파일을 찾을 수 없습니다: {self.concept_small_path}\n"
                f"prepare_concept_small.py를 먼저 실행하여 파일을 생성하세요."
            )
            return
        
        for chunk in self._read_csv_chunks(
            self.concept_small_path,
            self.CONCEPT_SMALL_COLUMNS,
            "CONCEPT_SMALL",
            chunk_size,
            skip_rows,
            max_rows
        ):
            cleaned = self.clean_concept_small_data(chunk)
            if len(cleaned) > 0:
                yield cleaned
