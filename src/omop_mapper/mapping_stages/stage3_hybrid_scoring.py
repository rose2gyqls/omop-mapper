"""
Stage 3: 최종 Semantic/Lexical 유사도 계산 및 Hybrid Score 산출
- 텍스트 유사도 (Lexical Similarity): N-gram 기반 Jaccard 유사도
- 의미적 유사도 (Semantic Similarity): SapBERT 임베딩 코사인 유사도
- 하이브리드 점수: 텍스트 유사도 40% + 의미적 유사도 60%
"""
from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np

try:
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class Stage3HybridScoring:
    """Stage 3: Hybrid Score 계산 및 최종 랭킹"""
    
    def __init__(
        self, 
        sapbert_model=None, 
        sapbert_tokenizer=None, 
        sapbert_device=None,
        text_weight: float = 0.4,
        semantic_weight: float = 0.6
    ):
        """
        Args:
            sapbert_model: SapBERT 모델
            sapbert_tokenizer: SapBERT 토크나이저
            sapbert_device: SapBERT 디바이스
            text_weight: 텍스트 유사도 가중치 (기본값: 0.4)
            semantic_weight: 의미적 유사도 가중치 (기본값: 0.6)
        """
        self.sapbert_model = sapbert_model
        self.sapbert_tokenizer = sapbert_tokenizer
        self.sapbert_device = sapbert_device
        self.text_weight = text_weight
        self.semantic_weight = semantic_weight
    
    def calculate_hybrid_scores(
        self, 
        entity_name: str,
        stage2_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Stage 2 후보들에 대해 Hybrid Score 계산 및 정렬
        
        Args:
            entity_name: 엔티티 이름
            stage2_candidates: Stage 2에서 수집된 Standard 후보들
            
        Returns:
            List[Dict]: Hybrid Score가 계산되고 정렬된 후보들
        """
        logger.info("=" * 80)
        logger.info("Stage 3: Semantic/Lexical 유사도 계산 및 Hybrid Score 산출")
        logger.info("=" * 80)
        
        final_candidates = []
        
        for i, candidate in enumerate(stage2_candidates, 1):
            concept = candidate['concept']
            elasticsearch_score = candidate['elasticsearch_score']
            search_type = candidate.get('search_type', 'unknown')
            is_original_standard = candidate.get('is_original_standard', True)
            
            logger.info(f"\n  [{i}/{len(stage2_candidates)}] {concept.get('concept_name', 'N/A')} "
                      f"(ID: {concept.get('concept_id', 'N/A')}) [{search_type}]")
            
            # non-std to std 변환을 거친 경우 텍스트 유사도를 0.9로 고정
            if not is_original_standard:
                logger.info(f"     ℹ️  Non-std to Std 변환 후보 - 텍스트 유사도 1.0으로 고정")
                text_sim = 1.0
                
                # 의미적 유사도만 계산
                semantic_sim = self._calculate_semantic_similarity(
                    entity_name, 
                    concept
                )
                
                # Hybrid Score 계산
                hybrid_score = (self.text_weight * text_sim) + \
                              (self.semantic_weight * semantic_sim)
                hybrid_score = max(0.0, min(1.0, hybrid_score))
            else:
                # 원래 Standard 후보는 정상적으로 Hybrid Score 계산
                hybrid_score, text_sim, semantic_sim = self._calculate_hybrid_score(
                    entity_name,
                    concept.get('concept_name', ''),
                    elasticsearch_score,
                    concept
                )
            
            logger.info(f"     📊 텍스트 유사도: {text_sim:.4f}")
            logger.info(f"     📊 의미적 유사도: {semantic_sim:.4f}")
            logger.info(f"     ⭐ 하이브리드 점수: {hybrid_score:.4f}")
            
            final_candidates.append({
                'concept': concept,
                'final_score': hybrid_score,
                'is_original_standard': candidate['is_original_standard'],
                'original_candidate': candidate['original_candidate'],
                'elasticsearch_score': elasticsearch_score,
                'text_similarity': text_sim,
                'semantic_similarity': semantic_sim,
                'search_type': search_type
            })
        
        # Hybrid Score 기준으로 정렬
        sorted_candidates = sorted(final_candidates, key=lambda x: x['final_score'], reverse=True)
        
        # 최종 순위 로깅
        logger.info("\n" + "=" * 80)
        logger.info("📊 Stage 3 결과 - Hybrid Score 순위:")
        logger.info("=" * 80)
        for i, candidate in enumerate(sorted_candidates[:10], 1):  # 상위 10개만 로깅
            concept = candidate['concept']
            search_type = candidate.get('search_type', 'unknown')
            logger.info(f"  {i}. {concept.get('concept_name', 'N/A')} "
                      f"(ID: {concept.get('concept_id', 'N/A')}) [{search_type}]")
            logger.info(f"     점수: {candidate['final_score']:.4f} "
                      f"(텍스트: {candidate['text_similarity']:.4f}, "
                      f"의미적: {candidate['semantic_similarity']:.4f})")
        
        logger.info("=" * 80)
        
        return sorted_candidates
    
    def _calculate_hybrid_score(
        self,
        entity_name: str,
        concept_name: str,
        elasticsearch_score: float,
        concept_source: Dict[str, Any]
    ) -> Tuple[float, float, float]:
        """
        텍스트 유사도와 의미적 유사도를 결합한 Hybrid Score 계산
        
        Args:
            entity_name: 엔티티 이름
            concept_name: 개념 이름
            elasticsearch_score: Elasticsearch 점수 (현재 미사용)
            concept_source: 개념 소스 데이터
            
        Returns:
            Tuple[float, float, float]: (Hybrid Score, 텍스트 유사도, 의미적 유사도)
        """
        try:
            # 1. 텍스트 유사도 계산 (Jaccard 유사도)
            text_similarity = self._calculate_text_similarity(entity_name, concept_name)
            
            # 2. 의미적 유사도 계산 (SapBERT 코사인 유사도)
            semantic_similarity = self._calculate_semantic_similarity(
                entity_name, 
                concept_source
            )
            
            # 3. Hybrid Score 계산
            hybrid_score = (self.text_weight * text_similarity) + \
                          (self.semantic_weight * semantic_similarity)
            
            # 점수를 0-1 범위로 제한
            hybrid_score = max(0.0, min(1.0, hybrid_score))
            text_similarity = max(0.0, min(1.0, text_similarity))
            semantic_similarity = max(0.0, min(1.0, semantic_similarity))
            
            return hybrid_score, text_similarity, semantic_similarity
            
        except Exception as e:
            logger.error(f"Hybrid Score 계산 실패: {e}")
            # 오류 발생시 텍스트 유사도만 사용
            fallback_similarity = self._calculate_text_similarity(entity_name, concept_name)
            return fallback_similarity, fallback_similarity, 0.0
    
    def _calculate_text_similarity(self, entity_name: str, concept_name: str) -> float:
        """
        두 문자열 간의 Jaccard 유사도 계산 (N-gram 3 기반)
        
        Args:
            entity_name: 원본 엔티티 이름
            concept_name: 비교할 개념 이름
            
        Returns:
            float: Jaccard 유사도 점수 (0.0 ~ 1.0)
        """
        if not entity_name or not concept_name:
            return 0.0
        
        # 대소문자 정규화
        entity_name = entity_name.lower()
        concept_name = concept_name.lower()
        
        # N-gram 3으로 분할
        entity_ngrams = self._get_ngrams(entity_name, n=3)
        concept_ngrams = self._get_ngrams(concept_name, n=3)
        
        if not entity_ngrams or not concept_ngrams:
            return 0.0
        
        # Jaccard 유사도 계산
        intersection = entity_ngrams.intersection(concept_ngrams)
        union = entity_ngrams.union(concept_ngrams)
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        
        return jaccard_similarity
    
    def _get_ngrams(self, text: str, n: int = 3) -> set:
        """
        텍스트를 N-gram으로 분할
        
        Args:
            text: 입력 텍스트
            n: N-gram 크기 (기본값: 3)
            
        Returns:
            set: N-gram 집합
        """
        if len(text) < n:
            return {text}
        
        ngrams = set()
        for i in range(len(text) - n + 1):
            ngrams.add(text[i:i + n])
        
        return ngrams
    
    def _calculate_semantic_similarity(
        self, 
        entity_name: str, 
        concept_source: Dict[str, Any]
    ) -> float:
        """
        SapBERT 임베딩을 사용한 의미적 유사도 계산
        
        Args:
            entity_name: 엔티티 이름
            concept_source: 개념 소스 데이터
            
        Returns:
            float: 의미적 유사도 (0.0 ~ 1.0)
        """
        if not HAS_SKLEARN:
            logger.debug("sklearn 미설치 - 의미적 유사도 0.0 사용")
            return 0.0
        
        concept_embedding = concept_source.get('concept_embedding')
        
        if not concept_embedding or len(concept_embedding) != 768:
            logger.debug(f"개념 임베딩 없음 - 의미적 유사도 0.0 사용: "
                        f"{concept_source.get('concept_name', 'N/A')}")
            return 0.0
        
        try:
            # 엔티티 임베딩 생성
            entity_embedding = self._get_entity_embedding(entity_name)
            
            if entity_embedding is None:
                logger.debug(f"엔티티 임베딩 생성 실패 - 의미적 유사도 0.0 사용")
                return 0.0
            
            # 코사인 유사도 계산
            concept_emb_array = np.array(concept_embedding).reshape(1, -1)
            entity_emb_array = entity_embedding.reshape(1, -1)
            semantic_similarity = cosine_similarity(entity_emb_array, concept_emb_array)[0][0]
            
            # 코사인 유사도는 -1~1 범위이므로 0~1로 정규화
            semantic_similarity = (semantic_similarity + 1.0) / 2.0
            
            logger.debug(f"의미적 유사도 계산 성공: {semantic_similarity:.4f} for "
                        f"{concept_source.get('concept_name', 'N/A')}")
            
            return semantic_similarity
            
        except Exception as e:
            logger.warning(f"의미적 유사도 계산 실패: {e}")
            return 0.0
    
    def _get_entity_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        SapBERT를 사용하여 텍스트의 임베딩 생성
        
        Args:
            text: 입력 텍스트
            
        Returns:
            Optional[np.ndarray]: 임베딩 벡터 또는 None
        """
        try:
            if self.sapbert_model is None or self.sapbert_tokenizer is None:
                return None
            
            import torch
            
            # 텍스트를 소문자로 변환
            text = text.lower().strip()
            
            # 토크나이징
            inputs = self.sapbert_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=25
            )
            inputs = {k: v.to(self.sapbert_device) for k, v in inputs.items()}
            
            # 임베딩 생성
            with torch.no_grad():
                outputs = self.sapbert_model(**inputs)
                # CLS 토큰의 임베딩 사용
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embedding.flatten()
            
        except Exception as e:
            logger.warning(f"SapBERT 임베딩 생성 실패: {e}")
            return None

