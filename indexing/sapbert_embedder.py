"""
SapBERT 임베딩 생성기 모듈

이 모듈은 SapBERT 모델을 사용하여 의료 엔터티명에 대한 임베딩을 생성합니다.
"""

import logging
import os
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional


class SapBERTEmbedder:
    """SapBERT 모델을 사용한 임베딩 생성기"""
    
    def __init__(
        self,
        model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        device: Optional[str] = None,
        max_length: int = 25,
        batch_size: int = 128,
        enabled: Optional[bool] = None
    ):
        """
        SapBERT 임베딩 생성기 초기화
        
        Args:
            model_name: 사용할 SapBERT 모델명
            device: 사용할 디바이스 (None일 경우 자동 선택)
            max_length: 토큰화 시 최대 길이
            batch_size: 배치 크기
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.enabled = True
        
        # # 임베딩 사용 여부 결정 (우선순위: 인자 > 환경변수 > 기본 False)
        # if enabled is None:
        #     env_val = os.getenv("OMOP_ENABLE_EMBEDDING", "0").strip()
        #     self.enabled = env_val in ("1", "true", "True", "YES", "yes")
        # else:
        #     self.enabled = bool(enabled)
        
        # 디바이스 설정
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # if not self.enabled:
        #     logging.info("SapBERT 임베딩 비활성화됨 (enabled=False). 모델 로딩을 건너뜁니다.")
        #     return
        
        logging.info(f"SapBERT 모델 로딩 중: {model_name}")
        logging.info(f"사용 디바이스: {self.device}")
        
        # 토크나이저와 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # 평가 모드로 설정
        
        logging.info("SapBERT 모델 로딩 완료")
    
    def encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        텍스트 리스트를 임베딩으로 변환
        
        Args:
            texts: 임베딩할 텍스트 리스트
            show_progress: 진행률 표시 여부
            
        Returns:
            임베딩 배열 (shape: [len(texts), embedding_dim])
        """
        if not self.enabled:
            logging.info("SapBERT 임베딩이 비활성화되어 빈 배열을 반환합니다.")
            return np.array([])
        
        if not texts:
            return np.array([])
            
        all_embeddings = []
        
        # 배치 단위로 처리
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="임베딩 생성 중")
            
        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + self.batch_size]
                batch_embeddings = self._encode_batch(batch_texts)
                all_embeddings.append(batch_embeddings)
        
        # 모든 배치 결과 결합
        embeddings = np.concatenate(all_embeddings, axis=0)
        logging.info(f"임베딩 생성 완료: {embeddings.shape}")
        
        return embeddings
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        배치 단위로 텍스트를 임베딩으로 변환
        
        Args:
            texts: 배치 텍스트 리스트
            
        Returns:
            배치 임베딩 배열
        """
        # 토큰화
        tokens = self.tokenizer.batch_encode_plus(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # GPU로 이동
        tokens_cuda = {k: v.to(self.device) for k, v in tokens.items()}
        
        # 모델 추론
        outputs = self.model(**tokens_cuda)
        
        # CLS 토큰 표현을 임베딩으로 사용
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        return cls_embeddings.cpu().numpy()
    
    def get_embedding_dimension(self) -> int:
        """임베딩 차원 수 반환"""
        if not self.enabled:
            return 0
        return self.model.config.hidden_size
    
    def __del__(self):
        """소멸자 - GPU 메모리 해제"""
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # 간단한 테스트
    logging.basicConfig(level=logging.INFO)
    # 테스트 시 임베딩을 강제로 켭니다. 실제 인덱싱 기본값은 비활성화입니다.
    embedder = SapBERTEmbedder(batch_size=4, enabled=True)
    test_texts = ["covid-19", "hypertension"]
    embeddings = embedder.encode_texts(test_texts)
    print(f"SapBERT 임베딩 테스트 완료: {embeddings.shape}")
    del embedder
