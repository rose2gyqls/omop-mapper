from .entity_mapping_api import (
    EntityMappingAPI,
    EntityInput,
    DomainID,
    MappingResult,
    map_single_entity
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
    
    # Utility functions
    "map_single_entity",
    
    # Supporting classes
    "ElasticsearchClient",
]