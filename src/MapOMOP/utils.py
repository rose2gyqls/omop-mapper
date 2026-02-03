"""
Common utility functions for OMOP Mapper.
"""

import math


def sigmoid_normalize(score: float, center: float = 3.0, scale: float = 1.0) -> float:
    """
    Normalize ES score to 0-1 range using sigmoid.
    
    Args:
        score: Raw ES score (e.g., BM25 score)
        center: Score at which sigmoid returns 0.5
        scale: Steepness of the curve
        
    Returns:
        Normalized score between 0 and 1
    """
    return 1 / (1 + math.exp(-(score - center) / scale))
