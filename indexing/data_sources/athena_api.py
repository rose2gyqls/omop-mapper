"""
Athena API Data Source

Reads OMOP CDM data from OHDSI Athena API.

Note: Athena API is not designed for bulk data download.
For large-scale indexing, download vocabulary files from https://athena.ohdsi.org
and use LocalCSVDataSource instead.
"""

import logging
import time
from typing import Iterator, Optional, List, Dict

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base import BaseDataSource, DataSourceType


class AthenaAPIDataSource(BaseDataSource):
    """Data source for OHDSI Athena API."""
    
    BASE_URL = "https://athena.ohdsi.org/api/v1"
    
    DEFAULT_VOCABULARIES = ["SNOMED", "ICD10CM", "ICD10", "RxNorm", "LOINC"]
    DEFAULT_DOMAINS = ["Condition", "Drug", "Measurement", "Procedure", "Observation"]
    
    def __init__(
        self,
        vocabularies: List[str] = None,
        domains: List[str] = None,
        standard_only: bool = True,
        page_size: int = 100,
        rate_limit_delay: float = 0.5
    ):
        """
        Initialize Athena API data source.
        
        Args:
            vocabularies: Vocabularies to fetch (default: common vocabularies)
            domains: Domains to fetch (default: common domains)
            standard_only: Only fetch standard concepts
            page_size: API page size
            rate_limit_delay: Delay between API calls (seconds)
        """
        super().__init__(DataSourceType.ATHENA_API)
        
        self.vocabularies = vocabularies or self.DEFAULT_VOCABULARIES
        self.domains = domains or self.DEFAULT_DOMAINS
        self.standard_only = standard_only
        self.page_size = page_size
        self.rate_limit_delay = rate_limit_delay
        
        # HTTP session with retry
        self.session = self._create_session()
        
        # Cached data
        self._concepts = None
        self._relationships = None
        self._synonyms = None
        
        self.logger.info(f"Initialized Athena API data source")
        self.logger.info(f"Vocabularies: {self.vocabularies}")
        self.logger.info(f"Domains: {self.domains}")
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry logic."""
        session = requests.Session()
        
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'MapOMOP/1.0'
        })
        
        return session
    
    def _api_get(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with rate limiting."""
        time.sleep(self.rate_limit_delay)
        
        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        return response.json()
    
    def _fetch_concepts(self) -> pd.DataFrame:
        """Fetch all concepts from API."""
        if self._concepts is not None:
            return self._concepts
        
        all_concepts = []
        
        for vocab in self.vocabularies:
            for domain in self.domains:
                self.logger.info(f"Fetching: {vocab}/{domain}")
                
                page = 0
                while True:
                    try:
                        params = {
                            'pageSize': self.page_size,
                            'page': page,
                            'vocabulary': vocab,
                            'domain': domain
                        }
                        if self.standard_only:
                            params['standardConcept'] = 'Standard'
                        
                        result = self._api_get('concepts', params)
                        items = result.get('content', [])
                        
                        if not items:
                            break
                        
                        for item in items:
                            all_concepts.append({
                                'concept_id': str(item.get('id', '')),
                                'concept_name': item.get('name', ''),
                                'domain_id': item.get('domain', {}).get('id', ''),
                                'vocabulary_id': item.get('vocabulary', {}).get('id', ''),
                                'concept_class_id': item.get('conceptClass', {}).get('id', ''),
                                'standard_concept': 'S' if item.get('standardConcept') == 'Standard' else None,
                                'concept_code': item.get('code', ''),
                                'valid_start_date': item.get('validStart', ''),
                                'valid_end_date': item.get('validEnd', ''),
                                'invalid_reason': item.get('invalidReason', '')
                            })
                        
                        if len(items) < self.page_size:
                            break
                        
                        page += 1
                        
                    except Exception as e:
                        self.logger.error(f"API error ({vocab}/{domain}): {e}")
                        break
        
        self._concepts = pd.DataFrame(all_concepts) if all_concepts else pd.DataFrame(columns=self.CONCEPT_COLUMNS)
        self.logger.info(f"Fetched {len(self._concepts)} concepts")
        
        return self._concepts
    
    def _fetch_relationships(self) -> pd.DataFrame:
        """Fetch relationships for cached concepts."""
        if self._relationships is not None:
            return self._relationships
        
        if self._concepts is None:
            self._fetch_concepts()
        
        all_relationships = []
        concept_ids = self._concepts['concept_id'].tolist() if len(self._concepts) > 0 else []
        
        for i, cid in enumerate(concept_ids):
            if i % 100 == 0:
                self.logger.info(f"Fetching relationships: {i}/{len(concept_ids)}")
            
            try:
                result = self._api_get(f'concepts/{cid}/relationships')
                
                for rel in result.get('items', []):
                    all_relationships.append({
                        'concept_id_1': str(cid),
                        'concept_id_2': str(rel.get('targetConceptId', '')),
                        'relationship_id': rel.get('relationshipId', ''),
                        'valid_start_date': rel.get('validStart', ''),
                        'valid_end_date': rel.get('validEnd', ''),
                        'invalid_reason': rel.get('invalidReason', '')
                    })
                    
            except Exception as e:
                self.logger.warning(f"Relationship fetch failed for {cid}: {e}")
        
        self._relationships = pd.DataFrame(all_relationships) if all_relationships else pd.DataFrame(columns=self.RELATIONSHIP_COLUMNS)
        self.logger.info(f"Fetched {len(self._relationships)} relationships")
        
        return self._relationships
    
    def _fetch_synonyms(self) -> pd.DataFrame:
        """Fetch synonyms for cached concepts."""
        if self._synonyms is not None:
            return self._synonyms
        
        if self._concepts is None:
            self._fetch_concepts()
        
        all_synonyms = []
        concept_ids = self._concepts['concept_id'].tolist() if len(self._concepts) > 0 else []
        
        for i, cid in enumerate(concept_ids):
            if i % 100 == 0:
                self.logger.info(f"Fetching synonyms: {i}/{len(concept_ids)}")
            
            try:
                result = self._api_get(f'concepts/{cid}')
                
                for syn in result.get('synonyms', []):
                    all_synonyms.append({
                        'concept_id': str(cid),
                        'concept_synonym_name': syn.get('name', ''),
                        'language_concept_id': str(syn.get('languageConceptId', ''))
                    })
                    
            except Exception as e:
                self.logger.warning(f"Synonym fetch failed for {cid}: {e}")
        
        self._synonyms = pd.DataFrame(all_synonyms) if all_synonyms else pd.DataFrame(columns=self.SYNONYM_COLUMNS)
        self.logger.info(f"Fetched {len(self._synonyms)} synonyms")
        
        return self._synonyms
    
    def get_concept_count(self) -> int:
        """Return total number of CONCEPT records."""
        return len(self._fetch_concepts())
    
    def get_relationship_count(self) -> int:
        """Return total number of CONCEPT_RELATIONSHIP records."""
        return len(self._fetch_relationships())
    
    def get_synonym_count(self) -> int:
        """Return total number of CONCEPT_SYNONYM records."""
        return len(self._fetch_synonyms())
    
    def _yield_chunks(
        self,
        df: pd.DataFrame,
        chunk_size: int,
        skip_rows: int,
        max_rows: Optional[int]
    ) -> Iterator[pd.DataFrame]:
        """Yield DataFrame in chunks."""
        if skip_rows > 0:
            df = df.iloc[skip_rows:]
        if max_rows is not None:
            df = df.iloc[:max_rows]
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size].copy()
            if len(chunk) > 0:
                yield chunk
    
    def read_concepts(
        self,
        chunk_size: int = 1000,
        skip_rows: int = 0,
        max_rows: Optional[int] = None
    ) -> Iterator[pd.DataFrame]:
        """Read CONCEPT data in chunks."""
        df = self._fetch_concepts()
        
        for chunk in self._yield_chunks(df, chunk_size, skip_rows, max_rows):
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
        df = self._fetch_relationships()
        
        for chunk in self._yield_chunks(df, chunk_size, skip_rows, max_rows):
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
        df = self._fetch_synonyms()
        
        for chunk in self._yield_chunks(df, chunk_size, skip_rows, max_rows):
            cleaned = self.clean_synonym_data(chunk)
            if len(cleaned) > 0:
                yield cleaned
    
    def clear_cache(self):
        """Clear cached data."""
        self._concepts = None
        self._relationships = None
        self._synonyms = None
        self.logger.info("Cache cleared")
