"""
SapBERT Embedder Module

Generates embeddings for medical entity names using the SapBERT model.
SapBERT is specifically designed for biomedical entity representation.

Supports dimension reduction via fixed random projection (768 → 128).
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

# Import shared projection utilities
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from MapOMOP.utils import reduce_embedding_dim, EMBEDDING_OUTPUT_DIM


class SapBERTEmbedder:
    """SapBERT-based text embedding generator."""
    
    DEFAULT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    
    def __init__(
        self,
        model_name: str = None,
        device: Optional[str] = None,
        max_length: int = 25,
        batch_size: int = 512,
        use_fp16: bool = True,
        output_dim: int = EMBEDDING_OUTPUT_DIM
    ):
        """
        Initialize SapBERT embedder.
        
        Args:
            model_name: HuggingFace model name (default: SapBERT-from-PubMedBERT-fulltext)
            device: Device to use ('cuda:0', 'cpu', etc.). Auto-detected if None.
            max_length: Maximum token length for tokenization
            batch_size: Batch size for encoding (use 512-1024 on GPU for throughput)
            use_fp16: Use half precision on GPU for ~2x speed (ignored on CPU)
            output_dim: Output embedding dimension (128 by default, None to skip reduction)
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.output_dim = output_dim
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self._is_cuda = isinstance(self.device, str) and "cuda" in self.device
        if self._is_cuda and not torch.cuda.is_available():
            self._is_cuda = False
        
        self.logger.info(f"Loading SapBERT model: {self.model_name}")
        self.logger.info(f"Device: {self.device}, batch_size: {batch_size}, fp16: {use_fp16 and self._is_cuda}")
        if self.output_dim:
            self.logger.info(f"Dimension reduction: 768 → {self.output_dim}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        if self._is_cuda and use_fp16:
            self.model = self.model.half()
        self.model.eval()
        
        self.logger.info("SapBERT model loaded successfully")
    
    @property
    def embedding_dim(self) -> int:
        """Get output embedding dimension (after projection if enabled)."""
        if self.output_dim:
            return self.output_dim
        return self.model.config.hidden_size
    
    def encode(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of texts to encode
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings (shape: [len(texts), embedding_dim])
        """
        if not texts:
            return np.array([])
        
        all_embeddings = []
        
        # Process in batches (inference_mode is faster than no_grad on modern PyTorch)
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings")
        
        with torch.inference_mode():
            for i in iterator:
                batch_texts = texts[i:i + self.batch_size]
                batch_embeddings = self._encode_batch(batch_texts)
                all_embeddings.append(batch_embeddings)
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Apply dimension reduction if configured
        if self.output_dim and self.output_dim < embeddings.shape[1]:
            embeddings = reduce_embedding_dim(embeddings, self.output_dim)
        
        if show_progress:
            self.logger.info(f"Generated embeddings: {embeddings.shape}")
        
        return embeddings
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a batch of texts.
        
        Args:
            texts: Batch of texts
            
        Returns:
            Batch embeddings as numpy array (float32 for ES compatibility)
        """
        # Tokenize
        tokens = self.tokenizer.batch_encode_plus(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        tokens = {k: v.to(self.device, non_blocking=True) for k, v in tokens.items()}
        
        # Get model output
        outputs = self.model(**tokens)
        
        # Use CLS token embedding
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Convert to float32 for Elasticsearch dense_vector; keep on CPU as numpy
        out = cls_embeddings.float().cpu().numpy()
        return out
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector
        """
        return self.encode([text], show_progress=False)[0]
    
    def cleanup(self):
        """Release resources and clear GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Resources cleaned up")
    
    def __del__(self):
        """Destructor to clean up resources."""
        self.cleanup()


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    embedder = SapBERTEmbedder(batch_size=4)
    
    test_texts = ["covid-19", "hypertension", "diabetes mellitus"]
    embeddings = embedder.encode(test_texts)
    
    print(f"Test completed: {embeddings.shape}")
    print(f"Embedding dimension: {embedder.embedding_dim}")
    
    embedder.cleanup()
