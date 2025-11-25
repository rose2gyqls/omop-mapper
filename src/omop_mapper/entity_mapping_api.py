from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from enum import Enum

from .elasticsearch_client import ElasticsearchClient
from .mapping_stages import (
    Stage1CandidateRetrieval,
    Stage2StandardCollection,
    Stage3HybridScoring
)
from .mapping_validation import MappingValidator

try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SAPBERT = True
except ImportError:
    HAS_SAPBERT = False

logger = logging.getLogger(__name__)


class DomainID(Enum):
    """도메인 ID (Elasticsearch에 저장된 형식과 일치)"""
    PROCEDURE = "Procedure"
    CONDITION = "Condition"
    DRUG = "Drug"
    OBSERVATION = "Observation"
    MEASUREMENT = "Measurement"
    THRESHOLD = "Threshold"
    DEMOGRAPHICS = "Demographics"
    PERIOD = "Period"
    PROVIDER = "Provider"
    DEVICE = "Device"


@dataclass
class EntityInput:
    """
    입력용 엔티티 데이터
    
    Args:
        entity_name: 매핑할 엔티티 이름
        domain_id: 도메인 ID (None이면 모든 도메인 검색, 지정하면 해당 도메인만 검색)
        vocabulary_id: 어휘체계 ID (선택사항)
    """
    entity_name: str
    domain_id: Optional[DomainID] = None
    vocabulary_id: Optional[str] = None


@dataclass
class MappingResult:
    """매핑 결과 데이터"""
    source_entity: EntityInput
    mapped_concept_id: str
    mapped_concept_name: str
    domain_id: str
    vocabulary_id: str
    concept_class_id: str
    standard_concept: str
    concept_code: str
    concept_embedding: List[float]
    valid_start_date: Optional[str] = None
    valid_end_date: Optional[str] = None
    invalid_reason: Optional[str] = None
    mapping_score: float = 0.0
    mapping_confidence: str = "low"
    mapping_method: str = "unknown"
    alternative_concepts: List[Dict[str, Any]] = None
    

class EntityMappingAPI:
    """엔티티 매핑 API 클래스 (3단계 매핑 파이프라인)"""

    def __init__(
        self,
        es_client: Optional[ElasticsearchClient] = None,
        confidence_threshold: float = 0.5
    ):
        """
        엔티티 매핑 API 초기화
        
        Args:
            es_client: Elasticsearch 클라이언트
            confidence_threshold: 매핑 신뢰도 임계치
        """
        self.es_client = es_client or ElasticsearchClient.create_default()
        self.confidence_threshold = confidence_threshold
        
        # SapBERT 모델 초기화 (지연 로딩)
        self._sapbert_model = None
        self._sapbert_tokenizer = None
        self._sapbert_device = None
        
        # Stage 모듈 초기화
        self.stage1 = Stage1CandidateRetrieval(
            es_client=self.es_client,
            has_sapbert=HAS_SAPBERT
        )
        
        self.stage2 = Stage2StandardCollection(
            es_client=self.es_client
        )
        
        self.stage3 = None  # SapBERT 모델 로딩 후 초기화
        
        # 검증 모듈 초기화
        self.validator = MappingValidator(
            es_client=self.es_client,
            openai_api_key=None,  # .env 파일에서 가져옴
            openai_model="gpt-4o-mini"
        )
        
        # 디버깅용 변수
        self._last_stage1_candidates = []
        self._last_stage2_candidates = []
        self._last_rerank_candidates = []
    
    def map_entity(self, entity_input: EntityInput) -> Optional[List[MappingResult]]:
        """
        단일 엔티티를 OMOP CDM에 3단계 매핑
        
        **도메인 검색 전략**:
        - entity_input.domain_id가 None이면: 6개 주요 도메인 모두에서 검색하여 최적 매핑 찾기
        - entity_input.domain_id가 지정되면: 해당 도메인에서만 검색 (특정 도메인 매핑이 필요한 경우)
        
        **3단계 매핑 파이프라인** (각 도메인별로 수행):
        - Stage 1: Elasticsearch에서 후보군 9개 추출 (Lexical 3 + Semantic 3 + Combined 3)
        - Stage 2: Non-standard to Standard 변환 및 중복 제거
        - Stage 3: LLM 기반 평가 및 최종 랭킹
        
        Args:
            entity_input: 매핑할 엔티티 정보
            
        Returns:
            List[MappingResult]: 각 도메인별 매핑 결과 리스트 (최고 점수 순으로 정렬)
        """
        try:
            entity_name = entity_input.entity_name
            input_domain = entity_input.domain_id
            
            # ===== 검색 대상 도메인 결정 =====
            if input_domain is None:
                # 케이스 1: 엔티티만 제공된 경우 → 모든 주요 도메인 검색
                target_domains = [
                    DomainID.DRUG,
                    DomainID.OBSERVATION,
                    DomainID.PROCEDURE,
                    DomainID.CONDITION,
                    DomainID.MEASUREMENT,
                    DomainID.DEVICE
                ]
                logger.info("=" * 100)
                logger.info(f"🚀 전체 도메인 3단계 엔티티 매핑 시작")
                logger.info(f"   엔티티: {entity_name}")
                logger.info(f"   검색 전략: 모든 도메인 검색 후 최적 매핑 선택")
                logger.info(f"   대상 도메인: Drug, Observation, Procedure, Condition, Measurement, Device (6개)")
                logger.info("=" * 100)
            else:
                # 케이스 2: 엔티티 + 도메인 제공된 경우 → 해당 도메인만 검색
                target_domains = [input_domain]
                logger.info("=" * 100)
                logger.info(f"🚀 단일 도메인 3단계 엔티티 매핑 시작")
                logger.info(f"   엔티티: {entity_name}")
                logger.info(f"   검색 전략: 지정된 도메인에서만 검색")
                logger.info(f"   대상 도메인: {input_domain.value} (1개)")
                logger.info("=" * 100)
            
            # SapBERT 모델 초기화 (필요시)
            if HAS_SAPBERT and self._sapbert_model is None:
                self._initialize_sapbert_model()
            
            # Stage 3 초기화 (SapBERT 모델 로딩 후)
            if self.stage3 is None:
                self.stage3 = Stage3HybridScoring(
                    sapbert_model=self._sapbert_model,
                    sapbert_tokenizer=self._sapbert_tokenizer,
                    sapbert_device=self._sapbert_device,
                    text_weight=0.4,
                    semantic_weight=0.6,
                    es_client=self.es_client,
                    openai_api_key=None,  # .env 파일에서 가져옴
                    openai_model="gpt-4o-mini"
                )
            
            # 엔티티 임베딩 생성
            entity_embedding = None
            if HAS_SAPBERT and self._sapbert_model is not None:
                entity_embedding = self._get_simple_embedding(entity_name)
                if entity_embedding is not None:
                    logger.info("✅ 엔티티 임베딩 생성 성공")
                else:
                    logger.warning("⚠️ 엔티티 임베딩 생성 실패")
            
            # 각 도메인별 매핑 결과 저장
            all_mapping_results = []
            self._last_domain_results = {}  # 도메인별 결과 저장
            self._all_domain_stage_results = {}  # 도메인별 Stage 결과 저장 (디버깅용)
            domain_candidates = {}  # 검색 도메인별 후보군 저장 (Best Domain 선택용)
            result_to_search_domain = {}  # 결과 객체 -> 검색 도메인 매핑
            
            # 각 도메인별로 Stage 1, 2, 3 수행
            for domain in target_domains:
                domain_result, domain_stages = self._map_entity_for_domain(
                    entity_name=entity_name,
                    domain_id=domain,
                    entity_embedding=entity_embedding,
                    entity_input=entity_input
                )
                
                if domain_result:
                    all_mapping_results.append(domain_result)
                    search_domain_str = str(domain.value)
                    
                    self._last_domain_results[search_domain_str] = domain_result
                    self._all_domain_stage_results[search_domain_str] = domain_stages
                    
                    # 도메인별 후보군 저장 (검색 도메인을 키로 사용)
                    if 'candidates' in domain_stages:
                        domain_candidates[search_domain_str] = domain_stages['candidates']
                    
                    # 결과 객체 -> 검색 도메인 매핑 저장 (나중에 Best 결과의 검색 도메인을 찾기 위함)
                    result_to_search_domain[id(domain_result)] = search_domain_str
            
            logger.info("\n" + "=" * 100)
            logger.info(f"✅ 도메인별 매핑 완료: {len(all_mapping_results)}개 도메인에서 결과 발견")
            logger.info("=" * 100)
            
            # 도메인별 최종 결과 요약 및 Best Domain의 후보군 설정
            if all_mapping_results:
                logger.info("\n📊 전체 도메인 최종 결과:")
                for idx, result in enumerate(all_mapping_results, 1):
                    logger.info(f"  {idx}. [{result.domain_id}] {result.mapped_concept_name} - 점수: {result.mapping_score:.4f}")
                
                best = max(all_mapping_results, key=lambda x: x.mapping_score)
                logger.info(f"\n🏆 최고 점수: [{best.domain_id}] {best.mapped_concept_name} ({best.mapping_score:.4f})")
                
                # Best result가 어느 검색 도메인에서 나왔는지 찾기
                best_search_domain = result_to_search_domain.get(id(best))
                
                if best_search_domain and best_search_domain in domain_candidates:
                    best_candidates = domain_candidates[best_search_domain]
                    self._last_stage1_candidates = best_candidates.get('stage1', [])
                    self._last_stage2_candidates = best_candidates.get('stage2', [])
                    self._last_rerank_candidates = best_candidates.get('stage3', [])
                    logger.info(f"✅ Best result의 검색 도메인 [{best_search_domain}]의 후보군을 디버깅 변수에 저장")
                    logger.info(f"   (결과 도메인: [{best.domain_id}])")
                else:
                    logger.warning(f"⚠️ Best result의 검색 도메인을 찾을 수 없음: {best.domain_id}")
            
            return all_mapping_results if all_mapping_results else None
                
        except Exception as e:
            logger.error(f"⚠️ 엔티티 매핑 오류: {str(e)}", exc_info=True)
            return None
    
    def _map_entity_for_domain(
        self,
        entity_name: str,
        domain_id,
        entity_embedding,
        entity_input: EntityInput
    ) -> tuple[Optional[MappingResult], Dict[str, Any]]:
        """
        특정 도메인에 대해 3단계 매핑 수행
        
        Args:
            entity_name: 엔티티 이름
            domain_id: 도메인 ID
            entity_embedding: 엔티티 임베딩
            entity_input: 원본 엔티티 입력
            
        Returns:
            tuple: (MappingResult, Stage 결과 딕셔너리) 또는 (None, {})
        """
        try:
            domain_str = str(domain_id.value) if hasattr(domain_id, 'value') else str(domain_id)
            
            logger.info("\n" + "=" * 100)
            logger.info(f"📍 도메인: {domain_str.upper()}")
            logger.info("=" * 100)
            
            # Stage별 결과 저장용
            stage_results = {
                'search_domain': domain_str,  # 검색한 도메인
                'result_domain': None,  # 실제 결과 도메인 (나중에 설정)
                'stage1_count': 0,
                'stage2_count': 0,
                'stage3_count': 0,
                'candidates': {}  # 후보군 정보 저장
            }
            
            # ===== Stage 1: 후보군 9개 추출 =====
            es_index = getattr(self.es_client, 'concept_index', 'concept')
            stage1_candidates = self.stage1.retrieve_candidates(
                entity_name=entity_name,
                domain_id=domain_str,
                entity_embedding=entity_embedding,
                es_index=es_index
            )
            
            if not stage1_candidates:
                logger.info(f"⚠️ [{domain_str}] Stage 1 - 검색 결과 없음")
                return None, {}
            
            stage_results['stage1_count'] = len(stage1_candidates)
            
            # ===== Stage 2: Standard 후보 수집 및 중복 제거 =====
            stage2_candidates = self.stage2.collect_standard_candidates(
                stage1_candidates=stage1_candidates,
                domain_id=domain_str
            )
            
            if not stage2_candidates:
                logger.info(f"⚠️ [{domain_str}] Stage 2 - Standard 후보 없음")
                return None, {}
            
            stage_results['stage2_count'] = len(stage2_candidates)
            
            # ===== Stage 3: LLM 기반 평가 =====
            stage3_candidates = self.stage3.calculate_hybrid_scores(
                entity_name=entity_name,
                stage2_candidates=stage2_candidates,
                stage1_candidates=stage1_candidates
            )
            
            if not stage3_candidates:
                logger.info(f"⚠️ [{domain_str}] Stage 3 - LLM 평가 실패")
                return None, {}
            
            stage_results['stage3_count'] = len(stage3_candidates)
            
            # ===== 최종 매핑 결과 생성 =====
            # entity_input의 domain_id를 현재 도메인으로 설정
            domain_entity_input = EntityInput(
                entity_name=entity_input.entity_name,
                domain_id=domain_id if isinstance(domain_id, DomainID) else None,
                vocabulary_id=entity_input.vocabulary_id
            )
            
            # LLM 방식 결과 사용
            mapping_result = self._create_final_mapping_result(domain_entity_input, stage3_candidates)
            
            # ===== 검증 단계 =====
            logger.info("\n" + "=" * 100)
            logger.info("🔍 매핑 검증 시작")
            logger.info("=" * 100)
            
            # 최종 매핑 결과 검증
            is_valid = self.validator.validate_mapping(
                entity_name=entity_name,
                concept_id=mapping_result.mapped_concept_id,
                concept_name=mapping_result.mapped_concept_name,
                synonyms=None  # Elasticsearch에서 조회
            )
            
            if not is_valid:
                logger.warning(f"⚠️ [{domain_str}] 최종 매핑 검증 실패: {mapping_result.mapped_concept_name}")
                logger.info("🔍 후보군 순차 검증 시작...")
                
                # 원래 후보 정보 저장
                original_candidate_id = mapping_result.mapped_concept_id
                original_candidate_name = mapping_result.mapped_concept_name
                
                # 후보군 순차 검증
                validated_candidate = self.validator.validate_candidates_sequentially(
                    entity_name=entity_name,
                    candidates=stage3_candidates,
                    max_candidates=10
                )
                
                if validated_candidate:
                    # 검증 통과한 후보로 매핑 결과 재생성
                    validated_concept_name = validated_candidate['concept'].get('concept_name', '')
                    logger.info(f"✅ 검증 통과한 후보 발견: {validated_concept_name}")
                    
                    # 검증 통과한 후보의 인덱스 찾기
                    validated_idx = None
                    for idx, candidate in enumerate(stage3_candidates):
                        if candidate['concept'].get('concept_id') == validated_candidate['concept'].get('concept_id'):
                            validated_idx = idx
                            break
                    
                    # 검증 통과한 후보를 맨 앞으로 이동하여 매핑 결과 재생성
                    if validated_idx is not None and validated_idx > 0:
                        reordered_candidates = [stage3_candidates[validated_idx]] + [
                            c for i, c in enumerate(stage3_candidates) if i != validated_idx
                        ]
                    else:
                        reordered_candidates = stage3_candidates
                    
                    mapping_result = self._create_final_mapping_result(domain_entity_input, reordered_candidates)
                    stage_results['validation_status'] = 'validated_alternative'
                    stage_results['original_candidate'] = {
                        'concept_id': original_candidate_id,
                        'concept_name': original_candidate_name
                    }
                    stage_results['validated_candidate'] = {
                        'concept_id': mapping_result.mapped_concept_id,
                        'concept_name': mapping_result.mapped_concept_name
                    }
                else:
                    logger.error(f"❌ [{domain_str}] 모든 후보 검증 실패 - 매핑 실패")
                    stage_results['validation_status'] = 'failed'
                    return None, stage_results
            else:
                logger.info(f"✅ [{domain_str}] 최종 매핑 검증 통과: {mapping_result.mapped_concept_name}")
                stage_results['validation_status'] = 'validated'
            
            # 실제 결과 도메인 저장
            stage_results['result_domain'] = mapping_result.domain_id
            
            logger.info(f"\n✅ [{domain_str}] 매핑 완료!")
            logger.info(f"   검색 도메인: {domain_str} → 결과 도메인: {mapping_result.domain_id}")
            logger.info(f"   개념: {mapping_result.mapped_concept_name} (ID: {mapping_result.mapped_concept_id})")
            logger.info(f"   점수: {mapping_result.mapping_score:.4f} | 신뢰도: {mapping_result.mapping_confidence}")
            logger.info(f"   Stage 경로: {stage_results['stage1_count']}개 → {stage_results['stage2_count']}개 → {stage_results['stage3_count']}개")
            logger.info(f"   검증 상태: {stage_results.get('validation_status', 'unknown')}")
            
            # 도메인별 후보군 정보를 stage_results에 저장
            stage_results['candidates'] = {
                'stage1': [
                    {
                        'concept_id': str(hit['_source'].get('concept_id', '')),
                        'concept_name': hit['_source'].get('concept_name', ''),
                        'domain_id': hit['_source'].get('domain_id', ''),
                        'vocabulary_id': hit['_source'].get('vocabulary_id', ''),
                        'standard_concept': hit['_source'].get('standard_concept', ''),
                        'elasticsearch_score': hit['_score'],
                        'search_type': hit.get('_search_type', 'unknown')
                    }
                    for hit in stage1_candidates
                ],
                'stage2': [
                    {
                        'concept_id': str(c['concept'].get('concept_id', '')),
                        'concept_name': c['concept'].get('concept_name', ''),
                        'domain_id': c['concept'].get('domain_id', ''),
                        'vocabulary_id': c['concept'].get('vocabulary_id', ''),
                        'standard_concept': c['concept'].get('standard_concept', ''),
                        'is_original_standard': c['is_original_standard'],
                        'search_type': c.get('search_type', 'unknown'),
                        'original_non_standard': c.get('original_non_standard', None)
                    }
                    for c in stage2_candidates
                ],
                'stage3': [
                    {
                        'concept_id': str(c['concept'].get('concept_id', '')),
                        'concept_name': c['concept'].get('concept_name', ''),
                        'domain_id': c['concept'].get('domain_id', ''),
                        'vocabulary_id': c['concept'].get('vocabulary_id', ''),
                        'standard_concept': c['concept'].get('standard_concept', ''),
                        'llm_score': c.get('llm_score', None),
                        'llm_rank': c.get('llm_rank', None),
                        'llm_reasoning': c.get('llm_reasoning', None),
                        'final_score': c.get('final_score', 0.0),
                        'search_type': c.get('search_type', 'unknown')
                    }
                    for c in stage3_candidates
                ]
            }
            
            return mapping_result, stage_results
            
        except Exception as e:
            logger.error(f"⚠️ [{domain_str}] 매핑 오류: {str(e)}", exc_info=True)
            return None, {}
    
    def _create_final_mapping_result(
        self, 
        entity_input: EntityInput, 
        sorted_candidates: List[Dict[str, Any]]
    ) -> MappingResult:
        """
        최종 매핑 결과 생성
        
        Args:
            entity_input: 원본 엔티티 입력
            sorted_candidates: 점수 순으로 정렬된 후보들
            
        Returns:
            MappingResult: 매핑 결과
        """
        best_candidate = sorted_candidates[0]
        alternative_candidates = sorted_candidates[1:4]  # 상위 3개 대안
        
        mapping_result = self._create_mapping_result(entity_input, best_candidate, alternative_candidates)
        
        mapping_type = "direct_standard" if best_candidate['is_original_standard'] else "non_standard_to_standard"
        logger.debug(f"매핑 유형: {mapping_type}")
        
        return mapping_result
    
    def _create_mapping_result(
        self, 
        entity_input: EntityInput, 
        best_candidate: Dict[str, Any], 
        alternative_candidates: List[Dict[str, Any]]
    ) -> MappingResult:
        """
        매핑 결과 생성
        
        Args:
            entity_input: 원본 엔티티 입력
            best_candidate: 최적 후보
            alternative_candidates: 대안 후보들
            
        Returns:
            MappingResult: 매핑 결과
        """
        concept = best_candidate['concept']
        final_score = best_candidate['final_score']
        
        # 대안 개념들 추출
        alternative_concepts = []
        for alt_candidate in alternative_candidates:
            if 'concept' in alt_candidate:
                alt_concept = alt_candidate['concept']
                alternative_concepts.append({
                    'concept_id': str(alt_concept.get('concept_id', '')),
                    'concept_name': alt_concept.get('concept_name', ''),
                    'vocabulary_id': alt_concept.get('vocabulary_id', ''),
                    'score': alt_candidate.get('final_score', 0)
                })
        
        # 매핑 방법 결정
        mapping_method = "direct_standard" if best_candidate['is_original_standard'] else "non_standard_to_standard"
        
        # 매핑 신뢰도 계산
        mapping_score = final_score
        mapping_confidence = self._determine_confidence(mapping_score)
        
        return MappingResult(
            source_entity=entity_input,
            mapped_concept_id=str(concept.get('concept_id', '')),
            mapped_concept_name=concept.get('concept_name', ''),
            domain_id=concept.get('domain_id', ''),
            vocabulary_id=concept.get('vocabulary_id', ''),
            concept_class_id=concept.get('concept_class_id', ''),
            standard_concept=concept.get('standard_concept', ''),
            concept_code=concept.get('concept_code', ''),
            valid_start_date=concept.get('valid_start_date'),
            valid_end_date=concept.get('valid_end_date'),
            invalid_reason=concept.get('invalid_reason'),
            concept_embedding=concept.get('concept_embedding'),
            mapping_score=mapping_score,
            mapping_confidence=mapping_confidence,
            mapping_method=mapping_method,
            alternative_concepts=alternative_concepts
        )
    
    def _determine_confidence(self, score: float) -> str:
        """
        매핑 신뢰도 결정 (0.0 ~ 1.0 점수 기준)
        
        신뢰도 기준:
        - 0.95 ~ 1.00: very_high (정확한 키워드 매칭)
        - 0.85 ~ 0.94: high (높은 유사도)
        - 0.70 ~ 0.84: medium (중간 유사도)
        - 0.50 ~ 0.69: low (낮은 유사도)
        - 0.00 ~ 0.49: very_low (매우 낮은 유사도)
        """
        if score >= 0.95:
            return "very_high"
        elif score >= 0.85:
            return "high"
        elif score >= 0.70:
            return "medium"
        elif score >= 0.50:
            return "low"
        else:
            return "very_low"
    
    def _get_simple_embedding(self, text: str):
        """
        SapBERT를 사용하여 텍스트의 임베딩 생성
        """
        try:
            # SapBERT 모델이 초기화되어 있는지 확인
            if not hasattr(self, '_sapbert_model') or self._sapbert_model is None:
                self._initialize_sapbert_model()
            
            if self._sapbert_model is None:
                return None
            
            # 텍스트를 소문자로 변환
            text = text.lower().strip()
            
            # 토크나이징
            inputs = self._sapbert_tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=25
            )
            inputs = {k: v.to(self._sapbert_device) for k, v in inputs.items()}
            
            # 임베딩 생성
            with torch.no_grad():
                outputs = self._sapbert_model(**inputs)
                # CLS 토큰의 임베딩 사용
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
            return embedding.flatten()
            
        except Exception as e:
            logger.warning(f"SapBERT 임베딩 생성 실패: {e}")
            return None
    
    def _initialize_sapbert_model(self):
        """SapBERT 모델 초기화 (지연 로딩)"""
        try:
            if not HAS_SAPBERT:
                logger.warning("SapBERT 관련 패키지가 설치되지 않음")
                self._sapbert_model = None
                self._sapbert_tokenizer = None
                self._sapbert_device = None
                return
            
            model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
            logger.info(f"🤖 SapBERT 모델 로딩 중: {model_name}")
            
            self._sapbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._sapbert_model = AutoModel.from_pretrained(model_name)
            
            # GPU 사용 가능 여부 확인
            if torch.cuda.is_available():
                self._sapbert_device = torch.device('cuda')
                logger.info(f"✅ GPU 사용 가능: {torch.cuda.get_device_name(0)}")
            else:
                self._sapbert_device = torch.device('cpu')
                logger.info("⚠️ GPU 사용 불가 - CPU 사용")
            
            self._sapbert_model.to(self._sapbert_device)
            self._sapbert_model.eval()
            
            logger.info(f"✅ SapBERT 모델 로딩 완료 (Device: {self._sapbert_device})")
            
        except Exception as e:
            logger.error(f"❌ SapBERT 모델 로딩 실패: {e}")
            self._sapbert_model = None
            self._sapbert_tokenizer = None
            self._sapbert_device = None
    
    def health_check(self) -> Dict[str, Any]:
        """API 상태 확인"""
        es_health = self.es_client.health_check()
        
        return {
            "api_status": "healthy",
            "elasticsearch_status": es_health,
            "confidence_threshold": self.confidence_threshold
        }
