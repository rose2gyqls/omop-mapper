"""
Stage 3: LLM 기반 후보군 평가 및 최종 랭킹
- OpenAI API를 사용하여 후보군 평가
- LLM 점수 기준으로 최종 순위 결정
"""
from typing import List, Dict, Any, Optional
import logging
import os
import json
from dotenv import load_dotenv

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger = logging.getLogger(__name__)
    logger.warning("openai가 설치되지 않았습니다. LLM 기능을 사용할 수 없습니다.")

if 'logger' not in locals():
    logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()


class Stage3HybridScoring:
    """Stage 3: LLM 기반 후보군 평가 및 최종 랭킹"""
    
    def __init__(
        self, 
        sapbert_model=None, 
        sapbert_tokenizer=None, 
        sapbert_device=None,
        text_weight: float = 0.4,
        semantic_weight: float = 0.6,
        es_client=None,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini"
    ):
        """
        Args:
            sapbert_model: SapBERT 모델 (사용하지 않지만 호환성을 위해 유지)
            sapbert_tokenizer: SapBERT 토크나이저 (사용하지 않지만 호환성을 위해 유지)
            sapbert_device: SapBERT 디바이스 (사용하지 않지만 호환성을 위해 유지)
            text_weight: 텍스트 유사도 가중치 (사용하지 않음)
            semantic_weight: 의미적 유사도 가중치 (사용하지 않음)
            es_client: Elasticsearch 클라이언트
            openai_api_key: OpenAI API 키 (None이면 .env 파일에서 가져옴)
            openai_model: OpenAI 모델명 (기본값: gpt-4o-mini)
        """
        self.es_client = es_client
        
        # OpenAI API 초기화
        self.openai_client = None
        self.openai_model = openai_model
        
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
    
    def calculate_hybrid_scores(
        self, 
        entity_name: str,
        stage2_candidates: List[Dict[str, Any]],
        stage1_candidates: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Stage 2 후보들에 대해 LLM 기반 평가 및 최종 랭킹
        
        **평가 방식**:
        - OpenAI GPT-4 모델을 사용하여 각 후보의 의미적 적합성 평가
        - 각 후보에 0.0~1.0 점수 부여
        - 하위 개념(sub-concept)으로 매핑되면 낮은 점수 부여
        - 최종 점수(final_score)는 LLM 점수(llm_score)를 사용
        
        Args:
            entity_name: 평가할 엔티티 이름
            stage2_candidates: Stage 2에서 수집된 Standard 후보들
            stage1_candidates: Stage 1 후보들 (사용하지 않음, 호환성 유지)
            
        Returns:
            List[Dict]: LLM 점수 기준으로 정렬된 후보들 (내림차순)
        """
        logger.info("=" * 80)
        logger.info("Stage 3: LLM 기반 후보군 평가 및 최종 랭킹")
        logger.info("=" * 80)
        
        if not self.openai_client:
            logger.error("⚠️ OpenAI API 클라이언트가 초기화되지 않았습니다.")
            return []
        
        if not stage2_candidates:
            logger.warning("⚠️ 평가할 후보가 없습니다.")
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
            llm_result = self._calculate_llm_scores(entity_name, final_candidates)
            
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
    
    def _calculate_llm_scores(
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
