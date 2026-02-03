"""
Elasticsearch Client Module

Provides connection and search functionality for OMOP CDM indices.
Supports concept, concept-relationship, and concept-synonym indices.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


class ElasticsearchClient:
    """Elasticsearch client for OMOP CDM data."""
    
    # Default configuration
    DEFAULT_HOST = "3.35.110.161"
    DEFAULT_PORT = 9200
    DEFAULT_USER = "elastic"
    DEFAULT_PASSWORD = "snomed"
    
    # Index names
    CONCEPT_INDEX = "concept-small"
    SYNONYM_INDEX = "concept-synonym"
    RELATIONSHIP_INDEX = "concept-relationship"
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_ssl: bool = False,
        timeout: int = 30
    ):
        """
        Initialize Elasticsearch client.
        
        Args:
            host: ES server host
            port: ES server port
            username: Username for authentication
            password: Password for authentication
            use_ssl: Whether to use SSL
            timeout: Request timeout in seconds
        """
        self.host = host or os.getenv('ES_SERVER_HOST', self.DEFAULT_HOST)
        self.port = port or int(os.getenv('ES_SERVER_PORT', str(self.DEFAULT_PORT)))
        self.username = username or os.getenv('ES_SERVER_USERNAME', self.DEFAULT_USER)
        self.password = password or os.getenv('ES_SERVER_PASSWORD', self.DEFAULT_PASSWORD)
        self.use_ssl = use_ssl
        self.timeout = timeout
        
        # Index names (can be overridden)
        self.concept_index = self.CONCEPT_INDEX
        self.concept_synonym_index = self.SYNONYM_INDEX
        self.concept_relationship_index = self.RELATIONSHIP_INDEX
        
        # Create client
        self.es_client = self._create_client()
        
        if self.es_client:
            logger.info(f"Elasticsearch client initialized: {self.host}:{self.port}")
        else:
            logger.warning("Elasticsearch client initialization failed")
    
    def _create_client(self) -> Optional[Elasticsearch]:
        """Create Elasticsearch client with retry logic."""
        scheme = "https" if self.use_ssl else "http"
        url = f"{scheme}://{self.host}:{self.port}"
            
        config = {
            'request_timeout': self.timeout,
            'max_retries': 3,
            'retry_on_timeout': True,
            'verify_certs': False,
            'ssl_show_warn': False
        }
        
        if self.username and self.password:
            config['basic_auth'] = (self.username, self.password)
        
        try:
            client = Elasticsearch(url, **config)
            
            if client.ping():
                logger.info(f"Connected to Elasticsearch: {url}")
                return client
            else:
                logger.warning(f"Elasticsearch ping failed: {url}")
                return client  # Return anyway, might work later
            
        except Exception as e:
            logger.error(f"Elasticsearch connection failed: {e}")
            return None
    
    def search_synonyms(self, concept_id: str) -> List[str]:
        """
        Search synonyms for a concept.
        
        Args:
            concept_id: OMOP concept ID
            
        Returns:
            List of synonym names
        """
        if not self.es_client:
            return []
        
        try:
            response = self.es_client.search(
                index=self.concept_synonym_index,
                body={
                    "query": {"term": {"concept_id": str(concept_id)}},
                    "size": 1000
                }
            )
            
            synonyms = []
            for hit in response.get('hits', {}).get('hits', []):
                name = hit['_source'].get('concept_synonym_name', '')
                if name and name not in synonyms:
                    synonyms.append(name)
            
            return synonyms
            
        except Exception as e:
            logger.error(f"Synonym search failed: {e}")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """Check cluster health status."""
        if not self.es_client:
            return {"status": "disconnected", "error": "Client not initialized"}
        
        try:
            if not self.es_client.ping():
                return {"status": "error", "error": "Ping failed"}
            
            info = self.es_client.info()
                    
            try:
                indices = self.es_client.cat.indices(format='json')
                index_count = len(indices)
            except Exception:
                index_count = "unknown"
                    
            return {
                "status": "connected",
                "cluster_name": info.get('cluster_name', 'unknown'),
                "version": info.get('version', {}).get('number', 'unknown'),
                "index_count": index_count,
                "host": self.host,
                "port": self.port
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def close(self):
        """Close connection."""
        if self.es_client:
            try:
                self.es_client.close()
                logger.info("Elasticsearch connection closed")
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
    
    @classmethod
    def create_default(cls) -> 'ElasticsearchClient':
        """Create client with default settings."""
        return cls()
