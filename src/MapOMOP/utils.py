"""
Common utility functions for OMOP Mapper.
"""

import math

import numpy as np


# =============================================================================
# Embedding dimension reduction
# =============================================================================

# Fixed projection config (must match between indexing and search)
EMBEDDING_INPUT_DIM = 768    # SapBERT hidden_size
EMBEDDING_OUTPUT_DIM = 128   # Reduced dimension for ES
PROJECTION_SEED = 42         # Fixed seed for reproducibility

_projection_matrix = None  # Lazy-loaded singleton


def get_projection_matrix(
    input_dim: int = EMBEDDING_INPUT_DIM,
    output_dim: int = EMBEDDING_OUTPUT_DIM,
    seed: int = PROJECTION_SEED
) -> np.ndarray:
    """
    Get deterministic random projection matrix (Gaussian).
    
    Same seed always produces the same matrix, so indexing and search
    are guaranteed to use identical projections.
    
    Args:
        input_dim: Original embedding dimension (768)
        output_dim: Target dimension (128)
        seed: Random seed for reproducibility
        
    Returns:
        Projection matrix of shape (input_dim, output_dim), float32
    """
    global _projection_matrix
    if _projection_matrix is not None:
        return _projection_matrix
    
    rng = np.random.RandomState(seed)
    matrix = rng.randn(input_dim, output_dim).astype(np.float32)
    # Scale by 1/sqrt(output_dim) for variance preservation (JL lemma)
    matrix /= np.sqrt(output_dim)
    _projection_matrix = matrix
    return _projection_matrix


def reduce_embedding_dim(
    embeddings: np.ndarray,
    output_dim: int = EMBEDDING_OUTPUT_DIM
) -> np.ndarray:
    """
    Reduce embedding dimensions via fixed random projection.
    
    Args:
        embeddings: Shape (N, 768) or (768,) for single vector
        output_dim: Target dimension
        
    Returns:
        Reduced embeddings, shape (N, output_dim) or (output_dim,)
    """
    proj = get_projection_matrix(output_dim=output_dim)
    
    single = embeddings.ndim == 1
    if single:
        embeddings = embeddings.reshape(1, -1)
    
    reduced = embeddings @ proj  # (N, 768) @ (768, 128) = (N, 128)
    
    # L2 normalize for cosine similarity
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    reduced = reduced / norms
    
    if single:
        return reduced.flatten()
    return reduced


# =============================================================================
# Score normalization
# =============================================================================

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
