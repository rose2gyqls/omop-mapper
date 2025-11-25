from .entity_mapping_api import (
    EntityMappingAPI,
    EntityInput,
    DomainID,
    MappingResult
)

from .elasticsearch_client import ElasticsearchClient

__version__ = "1.0.0"
__author__ = "rose"
__email__ = "hyobinkim@gmail.com"

__all__ = [
    # Main API classes
    "EntityMappingAPI",
    "EntityInput", 
    "DomainID",
    "MappingResult",
    
    # Supporting classes
    "ElasticsearchClient",
]