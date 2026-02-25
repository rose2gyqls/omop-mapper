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

# =============================================================================
# Candidate deduplication
# =============================================================================


def deduplicate_by_concept(
    candidates: list,
    *,
    get_concept=None,
    get_score=None
):
    """
    Deduplicate candidates by (concept_id, concept_name), keeping the one with higher score.

    Works with Stage 1 hits ({'_source': {...}}) and Stage 2/3 candidates ({'concept': {...}}).

    Args:
        candidates: List of candidate dicts
        get_concept: (candidate) -> dict with concept_id, concept_name.
            Default: extracts from candidate['concept'] or candidate['_source']
        get_score: (candidate) -> float for comparison (higher is better).
            Default: elasticsearch_score, or _score_normalized, or _score

    Returns:
        Deduplicated list (order not guaranteed)
    """
    if not candidates:
        return []

    def _default_get_concept(c):
        if 'concept' in c:
            return c['concept']
        if '_source' in c:
            return c['_source']
        return c

    def _default_get_score(c):
        return (
            c.get('elasticsearch_score') or
            c.get('_score_normalized') or
            c.get('_score', 0.0)
        )

    get_concept = get_concept or _default_get_concept
    get_score = get_score or _default_get_score

    unique = {}
    for c in candidates:
        conv = get_concept(c)
        key = (conv.get('concept_id', ''), conv.get('concept_name', ''))
        score = get_score(c)
        if key not in unique or score > get_score(unique[key]):
            unique[key] = c

    return list(unique.values())


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
