"""
Elasticsearch Client Module

Provides connection and search functionality for OMOP CDM indices.
Supports concept, concept-relationship, and concept-synonym indices.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

    from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


@dataclass 
class SearchResult:
    """Search result data class."""
    concept_id: str
    concept_name: str
    domain_id: str
    vocabulary_id: str
    concept_class_id: str
    standard_concept: str
    concept_code: str
    score: float
    synonyms: List[str] = field(default_factory=list)


class ElasticsearchClient:
    """Elasticsearch client for OMOP CDM data."""
    
    # Default configuration
    DEFAULT_HOST = "3.35.110.161"
    DEFAULT_PORT = 9200
    DEFAULT_USER = "elastic"
    DEFAULT_PASSWORD = "snomed"
    
    # Index names
    CONCEPT_INDEX = "concept"
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
    
    def search_concepts(
        self,
        query: str,
        domain_ids: Optional[List[str]] = None,
        vocabulary_ids: Optional[List[str]] = None,
        standard_concept_only: bool = False,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Search OMOP CDM concepts.
        
        Args:
            query: Search query text
            domain_ids: Domain ID filter (e.g., ['Condition', 'Drug'])
            vocabulary_ids: Vocabulary ID filter (e.g., ['SNOMED', 'RxNorm'])
            standard_concept_only: Only search standard concepts
            limit: Maximum results to return
            
        Returns:
            List of SearchResult objects
        """
        if not self.es_client:
            logger.warning("Elasticsearch client not initialized")
            return []
        
        try:
            indices = self._get_concept_indices()
            if not indices:
                logger.warning("No concept indices found")
                    return []
                
            search_body = self._build_search_query(
                    query, domain_ids, vocabulary_ids, standard_concept_only, limit
                )
                
                response = self.es_client.search(
                index=",".join(indices),
                    body=search_body
                )
            
            results = self._parse_search_results(response)
            logger.debug(f"Search '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _get_concept_indices(self) -> List[str]:
        """Get available concept indices."""
        try:
            indices = self.es_client.cat.indices(format='json')
            concept_indices = []
            
            for idx in indices:
                name = idx['index']
                if any(kw in name.lower() for kw in ['concept', 'omop']):
                    if 'synonym' not in name.lower() and 'relationship' not in name.lower():
                        concept_indices.append(name)
            
            return concept_indices if concept_indices else [self.concept_index]
            
        except Exception as e:
            logger.debug(f"Failed to list indices: {e}")
            return [self.concept_index]
    
    def _build_search_query(
        self,
        query: str,
        domain_ids: Optional[List[str]],
        vocabulary_ids: Optional[List[str]],
        standard_concept_only: bool,
        limit: int
    ) -> Dict[str, Any]:
        """Build Elasticsearch search query."""
        search_body = {
            "size": limit,
            "query": {
                "bool": {
                    "must": [{
                        "bool": {
                            "should": [
                                {"term": {"concept_name.keyword": {"value": query.lower(), "boost": 4.0}}},
                                {"match_phrase": {"concept_name": {"query": query, "boost": 2.5}}},
                                {"match": {"concept_name": {"query": query, "boost": 2.0}}},
                                {"match": {"concept_code": {"query": query, "boost": 2.0}}},
                                {"wildcard": {"concept_name": {"value": f"*{query}*", "boost": 1.5}}},
                                {"fuzzy": {"concept_name": {"value": query, "fuzziness": "AUTO", "boost": 1.0}}}
                            ],
                            "minimum_should_match": 1
                        }
                    }],
                    "filter": []
                }
            },
            "sort": [
                {"_score": {"order": "desc"}},
                {"concept_name.keyword": {"order": "asc"}}
            ]
        }
        
        # Domain filter
        if domain_ids:
            search_body["query"]["bool"]["filter"].append({
                "terms": {"domain_id": domain_ids}
            })
        
        # Vocabulary filter
        if vocabulary_ids:
            search_body["query"]["bool"]["filter"].append({
                "terms": {"vocabulary_id": vocabulary_ids}
            })
        
        # Standard concept filter
        if standard_concept_only:
            search_body["query"]["bool"]["filter"].append({
                "term": {"standard_concept": "S"}
            })
        
        return search_body
    
    def _parse_search_results(self, response: Dict[str, Any]) -> List[SearchResult]:
        """Parse Elasticsearch search response."""
        results = []
        
        for hit in response.get('hits', {}).get('hits', []):
            source = hit['_source']
            result = SearchResult(
                concept_id=str(source.get('concept_id', '')),
                concept_name=source.get('concept_name', ''),
                domain_id=source.get('domain_id', ''),
                vocabulary_id=source.get('vocabulary_id', ''),
                concept_class_id=source.get('concept_class_id', ''),
                standard_concept=source.get('standard_concept', ''),
                concept_code=source.get('concept_code', ''),
                score=hit['_score']
            )
            results.append(result)
        
        return results
    
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

    def search_synonyms_bulk(self, concept_ids: List[str]) -> Dict[str, List[str]]:
        """
        Bulk search synonyms for multiple concepts.
        
        Args:
            concept_ids: List of OMOP concept IDs
        
        Returns:
            Dict mapping concept_id to list of synonyms
        """
        result: Dict[str, List[str]] = {}
        
        if not self.es_client or not concept_ids:
            return result
        
        try:
            response = self.es_client.search(
                index=self.concept_synonym_index,
                body={
                    "query": {"terms": {"concept_id": [str(cid) for cid in concept_ids]}},
                    "size": max(100, len(concept_ids) * 10)
                }
            )
            
            for hit in response.get('hits', {}).get('hits', []):
                src = hit['_source']
                cid = str(src.get('concept_id', ''))
                syn = src.get('concept_synonym_name', '')
                
                if cid and syn:
                    if cid not in result:
                        result[cid] = []
                    if syn not in result[cid]:
                        result[cid].append(syn)
            
            return result
            
        except Exception as e:
            logger.error(f"Bulk synonym search failed: {e}")
            return {}
    
    def search_synonyms_with_embeddings_bulk(
        self,
        concept_ids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Bulk search synonyms with embeddings for multiple concepts.
        
        Args:
            concept_ids: List of OMOP concept IDs
        
        Returns:
            Dict mapping concept_id to list of synonym dicts with 'name' and 'embedding'
        """
        result: Dict[str, List[Dict[str, Any]]] = {}
        
        if not self.es_client or not concept_ids:
            return result
        
        try:
            response = self.es_client.search(
                index=self.concept_synonym_index,
                body={
                    "query": {"terms": {"concept_id": [str(cid) for cid in concept_ids]}},
                    "size": max(100, len(concept_ids) * 10)
                }
            )
            
            for hit in response.get('hits', {}).get('hits', []):
                src = hit['_source']
                cid = str(src.get('concept_id', ''))
                syn_name = src.get('concept_synonym_name', '')
                syn_embedding = src.get('concept_synonym_embedding')
                
                if cid and syn_name:
                    if cid not in result:
                        result[cid] = []
                    
                    entry = {'name': syn_name}
                    if syn_embedding and len(syn_embedding) == 768:
                        entry['embedding'] = syn_embedding
                    result[cid].append(entry)
            
            return result
            
        except Exception as e:
            logger.error(f"Bulk synonym+embedding search failed: {e}")
            return {}
    
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
