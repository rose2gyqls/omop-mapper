import json
import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

logger = logging.getLogger(__name__)

# OpenAI 라이브러리 임포트
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("openai가 설치되지 않았습니다. 검증 기능을 사용할 수 없습니다.")


class MappingValidator:
    def __init__(
        self,
        es_client=None,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini"
    ):
        self.es_client = es_client
        
        # OpenAI API 초기화
        self.openai_client = None
        self.openai_model = openai_model
        
        if not HAS_OPENAI:
            logger.error("⚠️ OpenAI 라이브러리가 설치되지 않았습니다. 검증 기능을 사용할 수 없습니다.")
            return
        
        try:
            api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
                logger.info(f"✅ MappingValidator 초기화 완료 (모델: {openai_model})")
            else:
                logger.error("⚠️ OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다. 검증 기능을 사용할 수 없습니다.")
        except Exception as e:
            logger.error(f"⚠️ OpenAI API 초기화 실패: {e}. 검증 기능을 사용할 수 없습니다.")
    
    def validate_mapping(
        self,
        entity_name: str,
        concept_id: str,
        concept_name: str,
        synonyms: Optional[List[str]] = None
    ) -> bool:
        """
        매핑 결과 검증 (True/False)
        
        Args:
            entity_name: 입력 엔티티 이름
            concept_id: 매핑된 concept ID
            concept_name: 매핑된 concept 이름
            synonyms: 동의어 리스트 (None이면 Elasticsearch에서 조회)
            
        Returns:
            bool: True면 매핑이 올바름, False면 매핑이 잘못됨
        """
        if not self.openai_client:
            logger.error("⚠️ OpenAI API 클라이언트가 초기화되지 않았습니다.")
            return False
        
        # 동의어가 제공되지 않으면 Elasticsearch에서 조회
        if synonyms is None:
            synonyms = self._fetch_synonyms(concept_id)
        
        # LLM을 통한 검증 수행
        try:
            result = self._validate_with_llm(entity_name, concept_id, concept_name, synonyms)
            return result
        except Exception as e:
            logger.error(f"⚠️ 검증 실패: {e}")
            return False
    
    def _fetch_synonyms(self, concept_id: str) -> List[str]:
        """
        Elasticsearch에서 동의어 조회
        
        Args:
            concept_id: concept ID
            
        Returns:
            List[str]: 동의어 리스트
        """
        if not self.es_client:
            logger.warning("⚠️ Elasticsearch 클라이언트가 없어 동의어를 조회할 수 없습니다.")
            return []
        
        try:
            synonyms = self.es_client.search_synonyms(str(concept_id))
            logger.debug(f"동의어 조회 완료: concept_id={concept_id}, 동의어 수={len(synonyms)}")
            return synonyms
        except Exception as e:
            logger.error(f"⚠️ 동의어 조회 실패: {e}")
            return []
    
    def _validate_with_llm(
        self,
        entity_name: str,
        concept_id: str,
        concept_name: str,
        synonyms: List[str]
    ) -> bool:
        """
        OpenAI LLM을 사용하여 매핑 검증
        
        Args:
            entity_name: 입력 엔티티 이름
            concept_id: 매핑된 concept ID
            concept_name: 매핑된 concept 이름
            synonyms: 동의어 리스트
            
        Returns:
            bool: True면 매핑이 올바름, False면 매핑이 잘못됨
        """
        # 프롬프트 생성
        prompt = self._create_validation_prompt(entity_name, concept_id, concept_name, synonyms)
        
        try:
            # OpenAI API 호출
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 의료 용어 매핑 검증 전문가입니다. 주어진 입력 엔티티와 매핑된 OMOP CDM concept의 일치 여부를 검증해야 합니다."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=256,
                response_format={"type": "json_object"}
            )
            
            # 응답 파싱
            response_text = response.choices[0].message.content
            result = self._parse_validation_response(response_text)
            return result
            
        except Exception as e:
            logger.error(f"OpenAI API 호출 실패: {e}")
            return False
    
    def _create_validation_prompt(
        self,
        entity_name: str,
        concept_id: str,
        concept_name: str,
        synonyms: List[str]
    ) -> str:
        """
        검증을 위한 프롬프트 생성
        
        Args:
            entity_name: 입력 엔티티 이름
            concept_id: 매핑된 concept ID
            concept_name: 매핑된 concept 이름
            synonyms: 동의어 리스트
            
        Returns:
            str: 프롬프트 문자열
        """
        # 동의어 처리 (세미콜론이나 슬래시로 구분된 경우 개별 동의어로 분리)
        all_synonyms = []
        for syn in synonyms:
            # 세미콜론으로 분리
            parts = syn.split(';')
            for part in parts:
                # 슬래시로 분리
                sub_parts = part.split('/')
                for sub_part in sub_parts:
                    cleaned = sub_part.strip()
                    if cleaned and cleaned not in all_synonyms:
                        all_synonyms.append(cleaned)
        
        # 동의어가 있으면 동의어를 힌트로 사용, 없으면 concept_name만 사용
        if all_synonyms:
            synonyms_text = "\n".join([f"- {syn}" for syn in all_synonyms[:50]])  # 최대 50개
            prompt = f"""다음 정보를 바탕으로 매핑이 올바른지 검증해주세요.

**입력 엔티티**: {entity_name}

**매핑된 Concept**:
- Concept ID: {concept_id}
- Concept Name: {concept_name}

**Concept의 동의어들** (힌트):
{synonyms_text}

**지시사항**:
1. 입력 엔티티({entity_name})가 매핑된 concept({concept_name})과 의미적으로 일치하는지 평가하세요.
2. 동의어들을 참고하여 입력 엔티티가 해당 concept으로 매핑되는 것이 적절한지 판단하세요.
3. 동의어 중 하나라도 입력 엔티티와 일치하거나 매우 유사하면 True로 판단하세요.
4. 입력 엔티티가 concept의 하위 개념이거나 완전히 다른 개념이면 False로 판단하세요.

**출력 형식** (JSON):
{{
  "is_valid": true 또는 false,
  "reasoning": "판단 이유 (한국어로 간단히)"
}}

JSON 형식으로만 응답해주세요. 다른 설명은 포함하지 마세요.
"""
        else:
            # 동의어가 없는 경우 concept_name과 entity_name만 비교
            prompt = f"""다음 정보를 바탕으로 매핑이 올바른지 검증해주세요.

**입력 엔티티**: {entity_name}

**매핑된 Concept**:
- Concept ID: {concept_id}
- Concept Name: {concept_name}

**지시사항**:
1. 입력 엔티티({entity_name})가 매핑된 concept({concept_name})과 의미적으로 일치하는지 평가하세요.
2. 입력 엔티티와 concept name이 의미적으로 일치하거나 매우 유사하면 True로 판단하세요.
3. 입력 엔티티가 concept의 하위 개념이거나 완전히 다른 개념이면 False로 판단하세요.

**출력 형식** (JSON):
{{
  "is_valid": true 또는 false,
  "reasoning": "판단 이유 (한국어로 간단히)"
}}

JSON 형식으로만 응답해주세요. 다른 설명은 포함하지 마세요.
"""
        
        return prompt
    
    def _parse_validation_response(self, response_text: str) -> bool:
        """
        LLM 응답 파싱
        
        Args:
            response_text: LLM 응답 텍스트
            
        Returns:
            bool: 검증 결과 (True/False)
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
            
            # is_valid 필드 추출
            is_valid = parsed.get('is_valid', False)
            reasoning = parsed.get('reasoning', '')
            
            logger.debug(f"검증 결과: is_valid={is_valid}, reasoning={reasoning}")
            
            return bool(is_valid)
            
        except Exception as e:
            logger.error(f"검증 응답 파싱 실패: {e}")
            logger.debug(f"응답 텍스트: {response_text[:500]}")
            return False
    
    def validate_candidates_sequentially(
        self,
        entity_name: str,
        candidates: List[Dict[str, Any]],
        max_candidates: int = 10
    ) -> Optional[Dict[str, Any]]:
        """
        후보군들을 순차적으로 검증하여 첫 번째 True 결과 반환
        
        Args:
            entity_name: 입력 엔티티 이름
            candidates: 검증할 후보군 리스트 (rank 순서대로 정렬되어 있어야 함)
            max_candidates: 최대 검증할 후보군 수
            
        Returns:
            Optional[Dict[str, Any]]: 검증 통과한 첫 번째 후보 또는 None
        """
        if not candidates:
            logger.warning("⚠️ 검증할 후보가 없습니다.")
            return None
        
        # 상위 후보들만 검증 (성능상 이유)
        top_candidates = candidates[:max_candidates]
        
        logger.info(f"🔍 순차 검증 시작: {len(top_candidates)}개 후보 검증")
        
        for idx, candidate in enumerate(top_candidates, 1):
            concept = candidate.get('concept', {})
            concept_id = str(concept.get('concept_id', ''))
            concept_name = concept.get('concept_name', '')
            
            if not concept_id or not concept_name:
                logger.warning(f"⚠️ 후보 {idx}: concept_id 또는 concept_name이 없습니다.")
                continue
            
            logger.info(f"  [{idx}/{len(top_candidates)}] 검증 중: {concept_name} (ID: {concept_id})")
            
            # 검증 수행
            is_valid = self.validate_mapping(
                entity_name=entity_name,
                concept_id=concept_id,
                concept_name=concept_name,
                synonyms=None  # Elasticsearch에서 조회
            )
            
            if is_valid:
                logger.info(f"  ✅ 검증 통과: {concept_name} (ID: {concept_id})")
                return candidate
            else:
                logger.info(f"  ❌ 검증 실패: {concept_name} (ID: {concept_id})")
        
        logger.warning(f"⚠️ 모든 후보({len(top_candidates)}개) 검증 실패")
        return None

