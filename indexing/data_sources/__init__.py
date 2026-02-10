"""
Data Sources Module

Provides data source adapters for different OMOP CDM sources:
    - LocalCSVDataSource: Read from local CSV files
    - PostgresDataSource: Read from PostgreSQL database
"""

from .base import BaseDataSource, DataSourceType
from .local_csv import LocalCSVDataSource
from .postgres import PostgresDataSource

__all__ = [
    'BaseDataSource',
    'DataSourceType',
    'LocalCSVDataSource',
    'PostgresDataSource'
]
