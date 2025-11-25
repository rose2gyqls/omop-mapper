"""
Stage 3: Hybrid 또는 LLM 기반 후보군 평가 및 최종 랭킹
- Hybrid: Text 유사도(Jaccard) + Semantic 유사도(SapBERT Cosine) 조합
- LLM: OpenAI API를 사용하여 후보군 평가
"""
from typing import List, Dict, Any, Optional
import logging
import os
import json
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Hybrid 모드용 라이브러리 임포트
try:
    import numpy as np
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_HYBRID_LIBS = True
except ImportError:
    HAS_HYBRID_LIBS = False
    np = None

    logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()


class Stage3HybridScoring:
    """Stage 3: Hybrid 또는 LLM 기반 후보군 평가 및 최종 랭킹"""
    
    def __init__(
        self, 
        sapbert_model=None, 
        sapbert_tokenizer=None, 
        sapbert_device=None,
        text_weight: float = 0.4,
        semantic_weight: float = 0.6,
        es_client=None,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
        scoring_mode: str = "llm"
    ):
        """
        Args:
            sapbert_model: SapBERT 모델 (hybrid 모드에서 사용)
            sapbert_tokenizer: SapBERT 토크나이저 (hybrid 모드에서 사용)
            sapbert_device: SapBERT 디바이스 (hybrid 모드에서 사용)
            text_weight: 텍스트 유사도 가중치 (hybrid 모드, 기본값: 0.4)
            semantic_weight: 의미적 유사도 가중치 (hybrid 모드, 기본값: 0.6)
            es_client: Elasticsearch 클라이언트
            openai_api_key: OpenAI API 키 (llm 모드, None이면 .env 파일에서 가져옴)
            openai_model: OpenAI 모델명 (llm 모드, 기본값: gpt-4o-mini)
            scoring_mode: 점수 계산 방식 ('llm' 또는 'hybrid', 기본값: 'llm')
        """
        self.es_client = es_client
        self.scoring_mode = scoring_mode.lower()
        
        # Hybrid 모드 설정
        self.sapbert_model = sapbert_model
        self.sapbert_tokenizer = sapbert_tokenizer
        self.sapbert_device = sapbert_device
        self.text_weight = text_weight
        self.semantic_weight = semantic_weight
        
        # OpenAI API 초기화 (LLM 모드)
        self.openai_client = None
        self.openai_model = openai_model
        
        if self.scoring_mode == "llm":
            if not HAS_OPENAI:
                logger.error("⚠️ OpenAI 라이브러리가 설치되지 않았습니다. LLM 기능을 사용할 수 없습니다.")
                return
            
            try:
                api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
                if api_key:
                    self.openai_client = OpenAI(api_key=api_key)
                    logger.info(f"✅ OpenAI API 초기화 완료 (모델: {openai_model})")
                else:
                    logger.error("⚠️ OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다. LLM 기능을 사용할 수 없습니다.")
            except Exception as e:
                logger.error(f"⚠️ OpenAI API 초기화 실패: {e}. LLM 기능을 사용할 수 없습니다.")
        elif self.scoring_mode == "hybrid":
            if not HAS_HYBRID_LIBS:
                logger.error("⚠️ Hybrid 모드에 필요한 라이브러리가 설치되지 않았습니다 (numpy, torch, sklearn).")
            elif sapbert_model is None or sapbert_tokenizer is None:
                logger.warning("⚠️ SapBERT 모델이 초기화되지 않았습니다. Hybrid 모드를 사용할 수 없습니다.")
            else:
                logger.info(f"✅ Hybrid 점수 계산 모드 초기화 (text: {text_weight}, semantic: {semantic_weight})")
        else:
            logger.error(f"⚠️ 알 수 없는 scoring_mode: {scoring_mode}. 'llm' 또는 'hybrid'를 사용하세요.")
    
    def calculate_hybrid_scores(
        self, 
        entity_name: str,
        stage2_candidates: List[Dict[str, Any]],
        stage1_candidates: Optional[List[Dict[str, Any]]] = None,
        entity_embedding: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Stage 2 후보들에 대해 Hybrid 또는 LLM 기반 평가 및 최종 랭킹
        
        **Hybrid 평가 방식** (scoring_mode='hybrid'):
        - 텍스트 유사도 (0.4): Jaccard 유사도 (n-gram=3)
          - Non-std to std 변환된 후보는 고정 0.9 점수
        - 의미적 유사도 (0.6): SapBERT 임베딩 + Cosine 유사도
        - 최종 점수 = 0.4 * text_similarity + 0.6 * semantic_similarity
        
        **LLM 평가 방식** (scoring_mode='llm'):
        - OpenAI GPT-4 모델을 사용하여 각 후보의 의미적 적합성 평가
        - 각 후보에 0.0~1.0 점수 부여
        - 하위 개념(sub-concept)으로 매핑되면 낮은 점수 부여
        - 최종 점수(final_score)는 LLM 점수(llm_score)를 사용
        
        Args:
            entity_name: 평가할 엔티티 이름
            stage2_candidates: Stage 2에서 수집된 Standard 후보들
            stage1_candidates: Stage 1 후보들 (사용하지 않음, 호환성 유지)
            entity_embedding: 엔티티의 SapBERT 임베딩 (hybrid 모드에서 사용)
            
        Returns:
            List[Dict]: 최종 점수 기준으로 정렬된 후보들 (내림차순)
        """
        logger.info("=" * 80)
        logger.info(f"Stage 3: {'Hybrid' if self.scoring_mode == 'hybrid' else 'LLM'} 기반 후보군 평가 및 최종 랭킹")
        logger.info("=" * 80)
        
        if not stage2_candidates:
            logger.warning("⚠️ 평가할 후보가 없습니다.")
            return []
        
        # Scoring mode에 따라 다른 방식 적용
        if self.scoring_mode == "hybrid":
            return self._calculate_hybrid_mode(entity_name, stage2_candidates, entity_embedding)
        elif self.scoring_mode == "llm":
            return self._calculate_llm_mode(entity_name, stage2_candidates)
        else:
            logger.error(f"⚠️ 알 수 없는 scoring_mode: {self.scoring_mode}")
            return []
    
    def _calculate_hybrid_mode(
        self,
        entity_name: str,
        stage2_candidates: List[Dict[str, Any]],
        entity_embedding: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid 모드: Text + Semantic 유사도 조합
        
        Args:
            entity_name: 엔티티 이름
            stage2_candidates: Stage 2 후보들
            entity_embedding: 엔티티의 SapBERT 임베딩
            
        Returns:
            List[Dict]: 최종 점수로 정렬된 후보들
        """
        if entity_embedding is None:
            logger.warning("⚠️ 엔티티 임베딩이 없습니다. 임베딩을 생성합니다.")
            entity_embedding = self._get_sapbert_embedding(entity_name)
        
        final_candidates = []
        
        for candidate in stage2_candidates:
            concept = candidate['concept']
            is_original_standard = candidate.get('is_original_standard', True)
            
            # 1. 텍스트 유사도 계산
            if is_original_standard:
                # 원래 Standard인 경우: Jaccard 유사도 계산
                text_similarity = self._calculate_jaccard_similarity(
                    entity_name, 
                    concept.get('concept_name', ''),
                    ngram=3
                )
            else:
                # Non-std to std 변환된 경우: 고정 0.9 점수
                text_similarity = 0.9
            
            # 2. 의미적 유사도 계산
            concept_embedding = concept.get('concept_embedding')
            if concept_embedding is not None and entity_embedding is not None and HAS_HYBRID_LIBS:
                # 임베딩을 numpy 배열로 변환
                if isinstance(concept_embedding, str):
                    # 문자열로 저장된 경우: JSON 파싱
                    try:
                        concept_embedding = np.array(json.loads(concept_embedding))
                    except:
                        concept_embedding = None
                elif isinstance(concept_embedding, list):
                    # 리스트로 저장된 경우: numpy 배열로 변환
                    try:
                        concept_embedding = np.array(concept_embedding)
                    except:
                        concept_embedding = None
                elif not isinstance(concept_embedding, np.ndarray):
                    # 그 외의 경우: numpy 배열로 시도
                    try:
                        concept_embedding = np.array(concept_embedding)
                    except:
                        concept_embedding = None
                
                # entity_embedding도 numpy 배열로 확인/변환
                if isinstance(entity_embedding, list):
                    try:
                        entity_embedding = np.array(entity_embedding)
                    except:
                        entity_embedding = None
                
                if concept_embedding is not None and entity_embedding is not None:
                    semantic_similarity = self._calculate_cosine_similarity(
                        entity_embedding,
                        concept_embedding
                    )
                else:
                    semantic_similarity = 0.0
            else:
                semantic_similarity = 0.0
                if concept_embedding is None:
                    logger.warning(f"⚠️ 후보 {concept.get('concept_id')}의 임베딩이 없습니다.")
            
            # 3. 최종 점수 계산: 0.4 * text + 0.6 * semantic
            final_score = (self.text_weight * text_similarity + 
                          self.semantic_weight * semantic_similarity)
            
            final_candidates.append({
                'concept': concept,
                'is_original_standard': is_original_standard,
                'original_candidate': candidate.get('original_candidate', {}),
                'elasticsearch_score': candidate.get('elasticsearch_score', 0.0),
                'search_type': candidate.get('search_type', 'unknown'),
                'text_similarity': text_similarity,
                'semantic_similarity': semantic_similarity,
                'final_score': final_score
            })
        
        # 최종 점수로 정렬
        sorted_candidates = sorted(
            final_candidates,
            key=lambda x: x['final_score'],
            reverse=True
        )
        
        # 결과 로깅
        logger.info("\n" + "=" * 80)
        logger.info("🔢 Stage 3 Hybrid 결과:")
        logger.info("=" * 80)
        for i, candidate in enumerate(sorted_candidates[:10], 1):
            concept = candidate['concept']
            search_type = candidate.get('search_type', 'unknown')
            is_std_marker = "✓" if candidate['is_original_standard'] else "→"
            logger.info(f"  {i}. [{search_type}] {is_std_marker} {concept.get('concept_name', 'N/A')} "
                       f"(ID: {concept.get('concept_id', 'N/A')})")
            logger.info(f"     텍스트: {candidate['text_similarity']:.4f}, "
                       f"의미적: {candidate['semantic_similarity']:.4f}, "
                       f"최종: {candidate['final_score']:.4f}")
        logger.info("=" * 80)
        
        return sorted_candidates
    
    def _calculate_llm_mode(
        self,
        entity_name: str,
        stage2_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        LLM 모드: OpenAI API를 사용한 평가
        
        Args:
            entity_name: 엔티티 이름
            stage2_candidates: Stage 2 후보들
            
        Returns:
            List[Dict]: LLM 점수로 정렬된 후보들
        """
        if not self.openai_client:
            logger.error("⚠️ OpenAI API 클라이언트가 초기화되지 않았습니다.")
            return []
        
        # 후보군 정보 준비
        final_candidates = []
        for candidate in stage2_candidates:
            concept = candidate['concept']
            final_candidates.append({
                'concept': concept,
                'is_original_standard': candidate.get('is_original_standard', True),
                'original_candidate': candidate.get('original_candidate', {}),
                'elasticsearch_score': candidate.get('elasticsearch_score', 0.0),
                'search_type': candidate.get('search_type', 'unknown')
            })
        
        # LLM 기반 평가 수행
        try:
            llm_result = self._calculate_llm_scores_api(entity_name, final_candidates)
            
            if not llm_result:
                logger.error("⚠️ LLM 평가 결과가 없습니다.")
                return []
            
            # LLM 점수를 각 후보에 추가
            for candidate in final_candidates:
                concept_id = str(candidate['concept'].get('concept_id', ''))
                if concept_id in llm_result:
                    candidate['llm_score'] = llm_result[concept_id]['score']
                    candidate['llm_rank'] = llm_result[concept_id]['rank']
                    candidate['llm_reasoning'] = llm_result[concept_id].get('reasoning', '')
                    # final_score를 llm_score로 설정 (최종 결과로 사용)
                    candidate['final_score'] = candidate['llm_score']
                else:
                    # LLM 평가에서 누락된 경우 점수 0.0
                    candidate['llm_score'] = 0.0
                    candidate['llm_rank'] = 999
                    candidate['llm_reasoning'] = 'LLM 평가에서 누락됨'
                    candidate['final_score'] = 0.0
            
            # LLM 점수 기준으로 정렬
            sorted_candidates = sorted(
                final_candidates, 
                key=lambda x: x.get('llm_score', 0.0), 
                reverse=True
            )
            
            # 최종 순위 로깅
            logger.info("\n" + "=" * 80)
            logger.info("🤖 Stage 3 LLM 결과 - OpenAI 순위:")
            logger.info("=" * 80)
            for i, candidate in enumerate(sorted_candidates[:10], 1):
                concept = candidate['concept']
                search_type = candidate.get('search_type', 'unknown')
                llm_score = candidate.get('llm_score', 0.0)
                llm_rank = candidate.get('llm_rank', 'N/A')
                logger.info(f"  {i}. {concept.get('concept_name', 'N/A')} "
                          f"(ID: {concept.get('concept_id', 'N/A')}) [{search_type}]")
                logger.info(f"     LLM 점수: {llm_score:.4f} (순위: {llm_rank})")
                if candidate.get('llm_reasoning'):
                    reasoning = candidate['llm_reasoning'][:100]
                    logger.info(f"     이유: {reasoning}...")
            logger.info("=" * 80)
            
            return sorted_candidates
            
        except Exception as e:
            logger.error(f"⚠️ LLM 평가 실패: {e}")
            return []
    
    def _calculate_jaccard_similarity(self, text1: str, text2: str, ngram: int = 3) -> float:
        """
        Jaccard 유사도 계산 (n-gram 기반)
        
        Args:
            text1: 첫 번째 텍스트
            text2: 두 번째 텍스트
            ngram: n-gram 크기 (기본값: 3)
            
        Returns:
            float: Jaccard 유사도 (0.0 ~ 1.0)
        """
        def get_ngrams(text: str, n: int) -> set:
            """텍스트에서 n-gram 추출"""
            text = text.lower().strip()
            if len(text) < n:
                return {text}
            return {text[i:i+n] for i in range(len(text) - n + 1)}
        
        ngrams1 = get_ngrams(text1, ngram)
        ngrams2 = get_ngrams(text2, ngram)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _get_sapbert_embedding(self, text: str) -> Optional[Any]:
        """
        텍스트의 SapBERT 임베딩 생성
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            np.ndarray: SapBERT 임베딩 벡터 (또는 None)
        """
        if not HAS_HYBRID_LIBS:
            logger.error("⚠️ Hybrid 모드 라이브러리가 없습니다.")
            return None
            
        if self.sapbert_model is None or self.sapbert_tokenizer is None:
            logger.warning("⚠️ SapBERT 모델이 초기화되지 않았습니다.")
            return None
        
        try:
            # 토크나이징
            inputs = self.sapbert_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            
            # 디바이스로 이동
            if self.sapbert_device:
                inputs = {k: v.to(self.sapbert_device) for k, v in inputs.items()}
            
            # 임베딩 생성
            with torch.no_grad():
                outputs = self.sapbert_model(**inputs)
                # CLS 토큰 임베딩 사용
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            return embedding
            
        except Exception as e:
            logger.error(f"SapBERT 임베딩 생성 실패: {e}")
            return None
    
    def _calculate_cosine_similarity(
        self, 
        embedding1: Any, 
        embedding2: Any
    ) -> float:
        """
        두 임베딩 간의 Cosine 유사도 계산
        
        Args:
            embedding1: 첫 번째 임베딩
            embedding2: 두 번째 임베딩
            
        Returns:
            float: Cosine 유사도 (0.0 ~ 1.0)
        """
        if not HAS_HYBRID_LIBS:
            logger.error("⚠️ Hybrid 모드 라이브러리가 없습니다.")
            return 0.0
            
        try:
            # 2D 배열로 변환 (cosine_similarity 요구사항)
            emb1 = embedding1.reshape(1, -1)
            emb2 = embedding2.reshape(1, -1)
            
            # Cosine 유사도 계산
            similarity = cosine_similarity(emb1, emb2)[0][0]
            
            # -1 ~ 1 범위를 0 ~ 1로 정규화
            normalized_similarity = (similarity + 1) / 2
            
            return float(normalized_similarity)
            
        except Exception as e:
            logger.error(f"Cosine 유사도 계산 실패: {e}")
            return 0.0
    
    def _calculate_llm_scores_api(
        self, 
        entity_name: str, 
        candidates: List[Dict[str, Any]],
        max_candidates: int = 15
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        OpenAI API를 사용하여 후보군 평가
        
        Args:
            entity_name: 엔티티 이름
            candidates: 평가할 후보군 리스트
            max_candidates: 평가할 최대 후보군 수 (기본값: 15)
            
        Returns:
            Dict[str, Dict[str, Any]]: concept_id를 키로 하는 평가 결과 딕셔너리
        """
        if not self.openai_client or not candidates:
            return None
        
        # 상위 후보만 평가 (성능상 이유)
        top_candidates = candidates[:max_candidates]
        
        # 프롬프트 생성
        prompt = self._create_llm_prompt(entity_name, top_candidates)
        
        try:
            # OpenAI API 호출
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 의료 용어 매핑 전문가입니다. 주어진 엔티티에 대해 가장 적합한 OMOP CDM 개념을 선택하고 각 후보에 대해 정확한 점수를 부여해야 합니다."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            
            # 응답 파싱
            response_text = response.choices[0].message.content
            result = self._parse_llm_response(response_text, top_candidates)
            return result
            
        except Exception as e:
            logger.error(f"OpenAI API 호출 실패: {e}")
            return None
    
    def _create_llm_prompt(self, entity_name: str, candidates: List[Dict[str, Any]]) -> str:
        """
        LLM을 위한 프롬프트 생성
        
        Args:
            entity_name: 엔티티 이름
            candidates: 후보군 리스트
            
        Returns:
            str: 프롬프트 문자열
        """
        candidates_info = []
        for i, candidate in enumerate(candidates, 1):
            concept = candidate['concept']
            candidates_info.append({
                'concept_id': str(concept.get('concept_id', '')),
                'concept_name': concept.get('concept_name', ''),
                'domain_id': concept.get('domain_id', '')
            })
        
        prompt = f"""다음 엔티티에 대해 가장 적합한 OMOP CDM 개념을 선택하고 각 후보에 대해 점수를 부여해주세요.

**엔티티 이름**: {entity_name}

**후보 개념들**:
{json.dumps(candidates_info, ensure_ascii=False, indent=2)}

**지시사항**:
1. 각 후보 개념이 엔티티 이름과 얼마나 의미적으로 일치하는지 평가하세요.
2. 의료 용어의 의미, 컨텍스트, 도메인 적합성을 고려하세요.
3. **중요**: 무조건 같은 레벨이거나 상위 레벨의 개념으로만 매핑되어야 합니다. 하위 개념(sub-concept)으로는 매핑되면 안 됩니다.
4. 각 후보에 대해 0.0~1.0 사이의 점수를 부여하세요 (1.0이 가장 적합함).
5. 선택 이유를 간단히 설명해주세요 (한국어로). 특히 하위 개념인 경우 이를 명확히 지적하고 점수를 낮게 부여하세요.

**출력 형식** (JSON):
{{
  "results": [
    {{
      "concept_id": "후보 개념 ID",
      "score": 0.0~1.0 사이의 점수,
      "rank": 1~{len(candidates)} 사이의 순위,
      "reasoning": "선택 이유 (한국어로 간단히)"
    }},
    ...
  ]
}}

JSON 형식으로만 응답해주세요. 다른 설명은 포함하지 마세요.
"""
        return prompt
    
    def _parse_llm_response(self, response_text: str, candidates: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        LLM 응답 파싱
        
        Args:
            response_text: LLM 응답 텍스트
            candidates: 후보군 리스트
            
        Returns:
            Dict[str, Dict[str, Any]]: concept_id를 키로 하는 평가 결과
        """
        try:
            # JSON 추출 (마크다운 코드 블록 제거)
            text = response_text.strip()
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()
            
            # JSON 파싱
            parsed = json.loads(text)
            
            # 결과 딕셔너리로 변환
            result = {}
            if 'results' in parsed:
                for item in parsed['results']:
                    concept_id = str(item.get('concept_id', ''))
                    if concept_id:
                        result[concept_id] = {
                            'score': float(item.get('score', 0.0)),
                            'rank': int(item.get('rank', 999)),
                            'reasoning': item.get('reasoning', '')
                        }
            
            # 모든 후보가 포함되었는지 확인 (없으면 점수 0.0으로 추가)
            for candidate in candidates:
                concept_id = str(candidate['concept'].get('concept_id', ''))
                if concept_id not in result:
                    result[concept_id] = {
                        'score': 0.0,
                        'rank': 999,
                        'reasoning': 'LLM 평가에서 누락됨'
                    }
            
            return result
            
        except Exception as e:
            logger.error(f"LLM 응답 파싱 실패: {e}")
            logger.debug(f"응답 텍스트: {response_text[:500]}")
            return {}
