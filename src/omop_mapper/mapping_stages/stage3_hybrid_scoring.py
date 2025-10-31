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
        semantic_weight: float = 0.6,
        es_client=None
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
        self.es_client = es_client
    
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
        
        # 후보들의 concept_id를 수집하여 동의어 임베딩을 일괄 조회
        synonyms_with_embeddings_map = {}
        try:
            if self.es_client:
                concept_ids = []
                for c in stage2_candidates:
                    cid = str(c['concept'].get('concept_id', ''))
                    if cid:
                        concept_ids.append(cid)
                concept_ids = list(dict.fromkeys(concept_ids))
                if hasattr(self.es_client, 'search_synonyms_with_embeddings_bulk'):
                    synonyms_with_embeddings_map = self.es_client.search_synonyms_with_embeddings_bulk(concept_ids)
        except Exception as e:
            logger.warning(f"동의어 임베딩 일괄 조회 실패: {e}")

        for i, candidate in enumerate(stage2_candidates, 1):
            concept = candidate['concept']
            elasticsearch_score = candidate['elasticsearch_score']
            search_type = candidate.get('search_type', 'unknown')
            is_original_standard = candidate.get('is_original_standard', True)
            
            logger.info(f"\n  [{i}/{len(stage2_candidates)}] {concept.get('concept_name', 'N/A')} "
                      f"(ID: {concept.get('concept_id', 'N/A')}) [{search_type}]")
            
            # non-std to std 변환을 거친 경우 텍스트 유사도 계산
            if not is_original_standard:
                text_sim = self._calculate_text_similarity(
                    entity_name,
                    concept.get('concept_name', '')
                )
                
                # Non-standard → Standard 변환이므로 기본 가점 추가
                text_sim = max(text_sim, 0.9)  # 최소 0.9 보장
                logger.info(f"     ℹ️  Non-std to Std 변환 후보 - 텍스트 유사도: {text_sim:.4f}")
                
                # 의미적 유사도 계산 (동의어 가점 포함)
                concept_id = str(concept.get('concept_id', ''))
                synonyms_with_embeddings = synonyms_with_embeddings_map.get(concept_id, [])
                semantic_sim, synonym_bonus = self._calculate_semantic_similarity_with_synonyms(
                    entity_name, 
                    concept,
                    synonyms_with_embeddings
                )
                
                if synonym_bonus > 0:
                    logger.info(f"     🎯 동의어 의미적 유사도 가점: +{synonym_bonus:.4f}")
                
                # Hybrid Score 계산
                hybrid_score = (self.text_weight * text_sim) + \
                              (self.semantic_weight * semantic_sim)
                hybrid_score = max(0.0, min(1.0, hybrid_score))
            else:
                # 원래 Standard 후보는 정상적으로 Hybrid Score 계산
                concept_id = str(concept.get('concept_id', ''))
                synonyms_with_embeddings = synonyms_with_embeddings_map.get(concept_id, [])
                hybrid_score, text_sim, semantic_sim, synonym_bonus = self._calculate_hybrid_score(
                    entity_name,
                    concept.get('concept_name', ''),
                    elasticsearch_score,
                    concept,
                    synonyms_with_embeddings
                )
                
                # 동의어 가점이 있으면 로깅
                if synonym_bonus > 0:
                    logger.info(f"     🎯 동의어 의미적 유사도 가점: +{synonym_bonus:.4f}")
            
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
        concept_source: Dict[str, Any],
        synonyms_with_embeddings: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[float, float, float, float]:
        """
        텍스트 유사도와 의미적 유사도를 결합한 Hybrid Score 계산
        
        Args:
            entity_name: 엔티티 이름
            concept_name: 개념 이름
            elasticsearch_score: Elasticsearch 점수 (현재 미사용)
            concept_source: 개념 소스 데이터
            synonyms_with_embeddings: 개념의 동의어 리스트 (임베딩 포함, 선택사항)
            
        Returns:
            Tuple[float, float, float, float]: (Hybrid Score, 텍스트 유사도, 의미적 유사도, 동의어 가점)
        """
        try:
            # 1. 텍스트 유사도 계산 (concept_name만 사용)
            text_similarity = self._calculate_text_similarity(entity_name, concept_name)
            
            # 2. 의미적 유사도 계산 (SapBERT 코사인 유사도, 동의어 가점 포함)
            semantic_similarity, synonym_bonus = self._calculate_semantic_similarity_with_synonyms(
                entity_name, 
                concept_source,
                synonyms_with_embeddings
            )
            
            # 3. Hybrid Score 계산 (동의어 가점은 의미적 유사도에 반영된 상태)
            hybrid_score = (self.text_weight * text_similarity) + \
                          (self.semantic_weight * semantic_similarity)
            
            # 점수를 0-1 범위로 제한
            hybrid_score = max(0.0, min(1.0, hybrid_score))
            text_similarity = max(0.0, min(1.0, text_similarity))
            semantic_similarity = max(0.0, min(1.0, semantic_similarity))
            
            return hybrid_score, text_similarity, semantic_similarity, synonym_bonus
            
        except Exception as e:
            logger.error(f"Hybrid Score 계산 실패: {e}")
            # 오류 발생시 텍스트 유사도만 사용
            fallback_similarity = self._calculate_text_similarity(entity_name, concept_name)
            return fallback_similarity, fallback_similarity, 0.0, 0.0
    
    def _normalize_text_for_similarity(self, text: str) -> str:
        """
        텍스트 유사도 계산을 위한 정규화
        - 하이픈, 언더스코어를 공백으로 변환
        - 연속된 공백을 하나로 통합
        - 앞뒤 공백 제거
        
        Args:
            text: 입력 텍스트
            
        Returns:
            str: 정규화된 텍스트
        """
        import re
        # 하이픈, 언더스코어를 공백으로 변환
        text = text.replace('-', ' ').replace('_', ' ')
        # 연속된 공백을 하나로 통합
        text = re.sub(r'\s+', ' ', text)
        # 앞뒤 공백 제거
        text = text.strip()
        return text
    
    def _calculate_text_similarity(self, entity_name: str, concept_name: str) -> float:
        """
        두 문자열 간의 유사도 계산 (N-gram Jaccard + 단어 단위 비교)
        특수문자(하이픈, 언더스코어 등) 정규화 후 계산
        
        Args:
            entity_name: 원본 엔티티 이름
            concept_name: 비교할 개념 이름
            
        Returns:
            float: 텍스트 유사도 점수 (0.0 ~ 1.0)
        """
        if not entity_name or not concept_name:
            return 0.0
        
        # 대소문자 정규화
        entity_name_lower = entity_name.lower()
        concept_name_lower = concept_name.lower()
        
        # 특수문자 정규화 (하이픈, 언더스코어를 공백으로)
        entity_name_normalized = self._normalize_text_for_similarity(entity_name_lower)
        concept_name_normalized = self._normalize_text_for_similarity(concept_name_lower)
        
        # 1. N-gram 3 기반 Jaccard 유사도
        entity_ngrams = self._get_ngrams(entity_name_normalized, n=3)
        concept_ngrams = self._get_ngrams(concept_name_normalized, n=3)
        
        if not entity_ngrams or not concept_ngrams:
            return 0.0
        
        intersection = entity_ngrams.intersection(concept_ngrams)
        union = entity_ngrams.union(concept_ngrams)
        ngram_similarity = len(intersection) / len(union) if union else 0.0
        
        # 2. 단어 단위 비교
        entity_words = set(entity_name_normalized.split())
        concept_words = set(concept_name_normalized.split())
        
        # 정확히 일치하는 단어
        exact_match_words = entity_words.intersection(concept_words)
        word_intersection = len(exact_match_words)
        word_union = len(entity_words.union(concept_words))
        word_jaccard = word_intersection / word_union if word_union else 0.0
        
        # 3. 부분 문자열 포함 보너스 (예: "cardiovascular" vs "vascular")
        partial_match_bonus = 0.0
        for entity_word in entity_words:
            for concept_word in concept_words:
                # 한 단어가 다른 단어에 포함되거나 매우 유사한 경우
                if entity_word in concept_word or concept_word in entity_word:
                    if entity_word != concept_word:  # 정확히 일치하면 이미 word_jaccard에 포함됨
                        partial_match_bonus += 0.1
                # 매우 유사한 단어 (편집 거리 기반, 예: "atherosclerotic" vs "arteriosclerotic")
                elif self._are_words_similar(entity_word, concept_word):
                    partial_match_bonus += 0.15
        
        # 부분 일치 보너스는 최대 0.3으로 제한
        partial_match_bonus = min(0.3, partial_match_bonus)
        
        # 4. 최종 유사도 = N-gram 유사도(60%) + 단어 Jaccard(30%) + 부분 일치 보너스(최대 10%)
        final_similarity = (0.6 * ngram_similarity) + (0.3 * word_jaccard) + min(0.1, partial_match_bonus)
        
        # 최대 1.0으로 제한
        return min(1.0, final_similarity)
    
    def _are_words_similar(self, word1: str, word2: str, threshold: float = 0.8) -> bool:
        """
        두 단어가 유사한지 판단 (편집 거리 기반)
        
        Args:
            word1: 첫 번째 단어
            word2: 두 번째 단어
            threshold: 유사도 임계값 (기본값: 0.8)
            
        Returns:
            bool: 유사하면 True
        """
        if not word1 or not word2:
            return False
        
        # 길이 차이가 너무 크면 유사하지 않음
        if abs(len(word1) - len(word2)) > max(len(word1), len(word2)) * 0.3:
            return False
        
        # N-gram 유사도로 판단
        w1_ngrams = self._get_ngrams(word1, n=3)
        w2_ngrams = self._get_ngrams(word2, n=3)
        
        if not w1_ngrams or not w2_ngrams:
            return False
        
        intersection = w1_ngrams.intersection(w2_ngrams)
        union = w1_ngrams.union(w2_ngrams)
        similarity = len(intersection) / len(union) if union else 0.0
        
        return similarity >= threshold
    
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
    
    def _calculate_semantic_similarity_with_synonyms(
        self, 
        entity_name: str, 
        concept_source: Dict[str, Any],
        synonyms_with_embeddings: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[float, float]:
        """
        SapBERT 임베딩을 사용한 의미적 유사도 계산 (동의어 임베딩 포함)
        
        Args:
            entity_name: 엔티티 이름
            concept_source: 개념 소스 데이터
            synonyms_with_embeddings: 동의어 리스트 (임베딩 포함, 선택사항)
                각 항목은 {'name': str, 'embedding': List[float]} 형태
            
        Returns:
            Tuple[float, float]: (의미적 유사도, 동의어 가점)
        """
        if not HAS_SKLEARN:
            logger.debug("sklearn 미설치 - 의미적 유사도 0.0 사용")
            return 0.0, 0.0
        
        concept_embedding = concept_source.get('concept_embedding')
        
        if not concept_embedding or len(concept_embedding) != 768:
            logger.debug(f"개념 임베딩 없음 - 의미적 유사도 0.0 사용: "
                        f"{concept_source.get('concept_name', 'N/A')}")
            return 0.0, 0.0
        
        try:
            # 엔티티 임베딩 생성
            entity_embedding = self._get_entity_embedding(entity_name)
            
            if entity_embedding is None:
                logger.debug(f"엔티티 임베딩 생성 실패 - 의미적 유사도 0.0 사용")
                return 0.0, 0.0
            
            # concept_name과의 코사인 유사도 계산
            concept_emb_array = np.array(concept_embedding).reshape(1, -1)
            entity_emb_array = entity_embedding.reshape(1, -1)
            base_semantic_similarity_raw = cosine_similarity(entity_emb_array, concept_emb_array)[0][0]
            
            # SapBERT 코사인 유사도는 보통 0.5~1.0 범위이므로 정규화 없이 사용
            # 음수 값이 나오는 경우를 대비해 0.0 이하는 0.0으로 제한
            base_semantic_similarity = max(0.0, base_semantic_similarity_raw)
            
            # 디버깅용 로깅 (debug 레벨로 변경하여 중복 방지)
            logger.debug(f"     코사인 유사도 (원본): {base_semantic_similarity_raw:.4f} → "
                       f"사용값: {base_semantic_similarity:.4f} "
                       f"(entity: '{entity_name}' vs concept: '{concept_source.get('concept_name', 'N/A')}')")
            
            # 동의어 임베딩과의 의미적 유사도 계산 (가점)
            synonym_bonus = 0.0
            best_synonym_sim = 0.0
            best_synonym_name = None
            
            if synonyms_with_embeddings:
                for syn_entry in synonyms_with_embeddings[:50]:  # 최대 50개 동의어만 확인
                    syn_name = syn_entry.get('name', '')
                    syn_embedding = syn_entry.get('embedding')
                    
                    if not syn_embedding or len(syn_embedding) != 768:
                        continue
                    
                    # 동의어 임베딩과 엔티티 임베딩의 코사인 유사도 계산
                    syn_emb_array = np.array(syn_embedding).reshape(1, -1)
                    syn_sim_raw = cosine_similarity(entity_emb_array, syn_emb_array)[0][0]
                    syn_sim = max(0.0, syn_sim_raw)  # 정규화 없이 사용, 음수는 0.0으로 제한
                    
                    if syn_sim > best_synonym_sim:
                        best_synonym_sim = syn_sim
                        best_synonym_name = syn_name
                
                # 동의어 의미적 유사도 기반 가점 부여
                # SapBERT 코사인 유사도 기준으로 재조정:
                # - 유사도 0.9 이상: +0.05 (매우 높은 유사도)
                # - 유사도 0.8 이상: +0.03 (높은 유사도)
                # - 유사도 0.7 이상: +0.01 (중간 유사도)
                if best_synonym_sim >= 0.9:
                    synonym_bonus = 0.05
                elif best_synonym_sim >= 0.8:
                    synonym_bonus = 0.03
                elif best_synonym_sim >= 0.7:
                    synonym_bonus = 0.01
                
                # 동의어 매칭이 있으면 로깅
                if synonym_bonus > 0:
                    logger.debug(f"     동의어 의미적 매칭: '{best_synonym_name}' "
                               f"(코사인 유사도: {best_synonym_sim:.4f}, 가점: +{synonym_bonus:.4f})")
            
            # 최종 의미적 유사도 = 기본 유사도 + 동의어 가점 (최대 1.0)
            final_semantic_similarity = min(1.0, base_semantic_similarity + synonym_bonus)
            
            logger.debug(f"의미적 유사도 계산: {final_semantic_similarity:.4f} "
                        f"(기본: {base_semantic_similarity:.4f}, 가점: +{synonym_bonus:.4f}) "
                        f"for {concept_source.get('concept_name', 'N/A')}")
            
            return final_semantic_similarity, synonym_bonus
            
        except Exception as e:
            logger.warning(f"의미적 유사도 계산 실패: {e}")
            return 0.0, 0.0
    
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

