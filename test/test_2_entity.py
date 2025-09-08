"""
엔티티 매핑 API 테스트 코드
두 개의 엔티티에 대해 6단계별로 후보군과 점수를 확인
"""

import sys
import os
import logging
import numpy as np
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# 프로젝트 루트의 src 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from omop_mapper.entity_mapping_api import (
    EntityMappingAPI, 
    EntityInput, 
    EntityTypeAPI
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EntityMappingTester:
    """엔티티 매핑 API 테스트 클래스"""
    
    def __init__(self):
        """테스트 초기화"""
        self.api = EntityMappingAPI()
        
        # SapBERT 모델 초기화
        self.model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        logger.info(f"🤖 SapBERT 모델 로딩 중: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"✅ SapBERT 모델 로딩 완료 (Device: {self.device})")
        except Exception as e:
            logger.error(f"❌ SapBERT 모델 로딩 실패: {e}")
            raise
            
        logger.info("✅ EntityMappingTester 초기화 완료")
    
    def _get_sapbert_embedding(self, text: str) -> np.ndarray:
        """
        SapBERT를 사용하여 텍스트의 임베딩을 생성
        
        Args:
            text: 임베딩을 생성할 텍스트
            
        Returns:
            임베딩 벡터 (768차원)
        """
        try:
            # 토크나이징
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 임베딩 생성
            with torch.no_grad():
                outputs = self.model(**inputs)
                # CLS 토큰의 임베딩 사용
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
            return embedding.flatten()
            
        except Exception as e:
            logger.error(f"SapBERT 임베딩 생성 실패: {e}")
            return np.zeros(768)  # 기본값 반환
    
    def _search_concepts_by_name(self, entity_name: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        concept_name이 일치하는 후보군을 찾는 간단한 검색
        
        Args:
            entity_name: 검색할 엔티티 이름
            top_k: 상위 K개 결과 반환
            
        Returns:
            List[매칭된 컨셉 후보들]
        """
        # concepts 인덱스에서 concept_name 기반 검색
        query = {
            "query": {
                "bool": {
                    "should": [
                        # 대소문자 무시 정확 일치
                        {
                            "term": {
                                "concept_name.keyword": {
                                    "value": entity_name.lower(),
                                    "boost": 9.0
                                }
                            }
                        },
                        # 부분 일치
                        {
                            "match": {
                                "concept_name": {
                                    "query": entity_name,
                                    "boost": 5.0
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": top_k,
            "sort": [
                {"_score": {"order": "desc"}},
                {"concept_name.keyword": {"order": "asc"}}
            ]
        }
        
        try:
            # concepts 인덱스에서 검색
            response = self.api.es_client.es_client.search(
                index="concepts",
                body=query
            )
            
            return response['hits']['hits'] if response['hits']['total']['value'] > 0 else []
            
        except Exception as e:
            logger.error(f"concept_name 검색 실패: {e}")
            return []
    
    def _search_concepts_hybrid(self, entity_name: str, top_k: int = 10, 
                               text_weight: float = 0.4, semantic_weight: float = 0.6) -> List[Dict[str, Any]]:
        """
        문자열 일치도와 의미적 유사도를 결합한 하이브리드 검색
        
        Args:
            entity_name: 검색할 엔티티 이름
            top_k: 상위 K개 결과 반환
            text_weight: 문자열 일치도 가중치
            semantic_weight: 의미적 유사도 가중치
            
        Returns:
            List[하이브리드 점수로 정렬된 컨셉 후보들]
        """
        logger.info(f"🔍 하이브리드 검색 시작: '{entity_name}' (텍스트:{text_weight}, 의미:{semantic_weight})")
        
        # 1단계: 엔티티 임베딩 생성
        entity_embedding = self._get_sapbert_embedding(entity_name)
        logger.info(f"📊 엔티티 임베딩 생성 완료 (Shape: {entity_embedding.shape})")
        
        # 2단계: 문자열 기반 검색으로 초기 후보 확보
        text_candidates = self._search_concepts_by_name(entity_name, top_k=50)  # 더 많은 후보 확보
        logger.info(f"📝 문자열 검색 결과: {len(text_candidates)}개 후보")
        
        if not text_candidates:
            logger.warning("문자열 검색 결과 없음")
            return []
        
        # 3단계: 각 후보에 대해 하이브리드 점수 계산
        hybrid_candidates = []
        
        for candidate in text_candidates:
            try:
                source = candidate['_source']
                concept_name = source.get('concept_name', '')
                elasticsearch_score = candidate['_score']
                
                # 문자열 유사도 (정규화된 Elasticsearch 점수)
                max_es_score = text_candidates[0]['_score'] if text_candidates else 1.0
                text_similarity = elasticsearch_score / max_es_score
                
                # 의미적 유사도 계산
                concept_embedding = source.get('concept_embedding')
                if concept_embedding and len(concept_embedding) == 768:
                    concept_emb_array = np.array(concept_embedding).reshape(1, -1)
                    entity_emb_array = entity_embedding.reshape(1, -1)
                    semantic_similarity = cosine_similarity(entity_emb_array, concept_emb_array)[0][0]
                else:
                    # 임베딩이 없는 경우 문자열 유사도로 대체
                    semantic_similarity = self.api._calculate_similarity(entity_name, concept_name)
                    logger.debug(f"임베딩 없음 - 문자열 유사도 사용: {concept_name}")
                
                # 하이브리드 점수 계산
                hybrid_score = (text_weight * text_similarity) + (semantic_weight * semantic_similarity)
                
                # 후보 정보 저장
                hybrid_candidate = {
                    '_source': source,
                    '_score': elasticsearch_score,
                    'text_similarity': text_similarity,
                    'semantic_similarity': semantic_similarity,
                    'hybrid_score': hybrid_score,
                    'original_candidate': candidate
                }
                
                hybrid_candidates.append(hybrid_candidate)
                
            except Exception as e:
                logger.error(f"하이브리드 점수 계산 실패: {e}")
                continue
        
        # 4단계: 하이브리드 점수로 정렬
        hybrid_candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # 상위 K개 반환
        top_candidates = hybrid_candidates[:top_k]
        
        logger.info(f"🎯 하이브리드 검색 완료: 상위 {len(top_candidates)}개 후보 반환")
        
        # 점수 정보 로깅
        for i, candidate in enumerate(top_candidates[:3], 1):
            source = candidate['_source']
            logger.info(f"  {i}. {source.get('concept_name', 'N/A')}")
            logger.info(f"     텍스트: {candidate['text_similarity']:.3f}, "
                       f"의미: {candidate['semantic_similarity']:.3f}, "
                       f"하이브리드: {candidate['hybrid_score']:.3f}")
        
        return top_candidates
    
    def test_entity_mapping_6_steps(self, entity_name: str, entity_type: str, 
                                   golden_concept_id: str, golden_concept_name: str) -> None:
        """
        엔티티 매핑을 6단계별로 상세 테스트
        
        Args:
            entity_name: 테스트할 엔티티 이름
            entity_type: 엔티티 타입
            golden_concept_id: 골든셋 컨셉 ID
            golden_concept_name: 골든셋 컨셉 이름
        """
        print(f"\n{'='*80}")
        print(f"🔍 엔티티: {entity_name}")
        print(f"🎯 골든셋: {golden_concept_id} - {golden_concept_name}")
        print(f"{'='*80}")
        
        # 엔티티 이름 전처리
        preprocessed_name = self.api._preprocess_entity_name(entity_name)
        print(f"📝 전처리: '{entity_name}' → '{preprocessed_name}'")
        
        # 엔티티 입력 생성
        entity_input = EntityInput(
            entity_name=preprocessed_name,
            entity_type=EntityTypeAPI(entity_type),
            confidence=1.0
        )
        
        # 사전 매핑 정보 준비
        entities_to_map = self.api._prepare_entity_for_mapping(entity_input)
        if not entities_to_map:
            print("❌ 엔티티 매핑 준비 실패")
            return
        
        entity_info = entities_to_map[0]
        
        # ===== 1단계: 하이브리드 검색 → 문자열 일치도 + 의미적 유사도 =====
        print(f"\n🔍 1단계: 하이브리드 검색 → 문자열 일치도 + 의미적 유사도")
        print(f"{'='*60}")
        
        candidates = self._search_concepts_hybrid(preprocessed_name, top_k=5, text_weight=0.4, semantic_weight=0.6)
        
        if not candidates:
            print("❌ 검색 결과 없음")
            return
        
        print(f"총 {len(candidates)}개 후보 발견:")
        for i, candidate in enumerate(candidates, 1):
            source = candidate['_source']
            hybrid_score = candidate.get('hybrid_score', 0.0)
            text_sim = candidate.get('text_similarity', 0.0)
            semantic_sim = candidate.get('semantic_similarity', 0.0)
            
            print(f"  {i}. 하이브리드 점수: {hybrid_score:.3f}")
            print(f"     └ 텍스트 유사도: {text_sim:.3f} | 의미적 유사도: {semantic_sim:.3f}")
            print(f"     컨셉 ID: {source.get('concept_id', 'N/A')}")
            print(f"     컨셉명: {source.get('concept_name', 'N/A')}")
            print(f"     도메인: {source.get('domain_id', 'N/A')}")
            print(f"     어휘체계: {source.get('vocabulary_id', 'N/A')}")
            print(f"     표준여부: {source.get('standard_concept', 'N/A')}")
            print()
        
        # ===== 2단계: Standard/Non-standard 분류 =====
        print(f"🔄 2단계: Standard/Non-standard 분류")
        print(f"{'='*60}")
        
        standard_candidates = []
        non_standard_candidates = []
        
        for i, candidate in enumerate(candidates, 1):
            source = candidate['_source']
            # 하이브리드 후보에서는 original_candidate가 있을 수 있음
            original_candidate = candidate.get('original_candidate', candidate)
            
            if source.get('standard_concept') == 'S':
                standard_candidates.append(original_candidate)
                print(f"  {i}. ✅ Standard: {source.get('concept_name', 'N/A')}")
            else:
                non_standard_candidates.append(original_candidate)
                print(f"  {i}. ⚠️ Non-standard: {source.get('concept_name', 'N/A')}")
        
        print(f"\n  📊 분류 결과: Standard {len(standard_candidates)}개, Non-standard {len(non_standard_candidates)}개")
        
        # ===== 3단계: Non-standard인 경우 → Maps to 관계로 Standard 후보 조회 =====
        print(f"\n🔗 3단계: Non-standard → Maps to 관계로 Standard 후보 조회")
        print(f"{'='*60}")
        
        all_standard_candidates = []
        
        # Standard 후보들은 그대로 사용
        for candidate in standard_candidates:
            source = candidate['_source']
            all_standard_candidates.append({
                'concept': source,
                'final_score': candidate['_score'],
                'is_original_standard': True,
                'original_candidate': candidate
            })
        
        # Non-standard 후보들은 Standard 후보 조회
        for i, candidate in enumerate(non_standard_candidates, 1):
            source = candidate['_source']
            concept_id = str(source.get('concept_id', ''))
            print(f"  Non-standard {i}: {source.get('concept_name', 'N/A')} (ID: {concept_id})")
            
            standard_candidates_from_non = self.api._get_standard_candidates(concept_id, entity_info["domain_id"])
            
            if standard_candidates_from_non:
                print(f"    → Maps to 관계로 {len(standard_candidates_from_non)}개 Standard 후보 발견")
                for j, std_candidate in enumerate(standard_candidates_from_non[:3], 1):  # 상위 3개만 출력
                    print(f"      {j}. {std_candidate.get('concept_name', 'N/A')} (ID: {std_candidate.get('concept_id', 'N/A')})")
            else:
                print(f"    → Maps to 관계 없음")
            
            print()
        
        # ===== 4단계: Non-standard → Standard 후보 조회 및 임시 저장 =====
        print(f"🔗 4단계: Non-standard → Standard 후보 조회 및 임시 저장")
        print(f"{'='*60}")
        
        # Non-standard 후보들의 Standard 후보들을 임시로 저장
        non_standard_to_standard_mappings = []
        
        for i, candidate in enumerate(non_standard_candidates, 1):
            source = candidate['_source']
            concept_id = str(source.get('concept_id', ''))
            print(f"  Non-standard {i}: {source.get('concept_name', 'N/A')} (ID: {concept_id})")
            
            standard_candidates_from_non = self.api._get_standard_candidates(concept_id, entity_info["domain_id"])
            
            if standard_candidates_from_non:
                print(f"    → Maps to 관계로 {len(standard_candidates_from_non)}개 Standard 후보 발견")
                for j, std_candidate in enumerate(standard_candidates_from_non[:3], 1):  # 상위 3개만 출력
                    print(f"      {j}. {std_candidate.get('concept_name', 'N/A')} (ID: {std_candidate.get('concept_id', 'N/A')}")
                
                # 모든 Standard 후보들을 임시 저장 (나중에 유사도 재계산)
                non_standard_to_standard_mappings.append({
                    'non_standard_source': source,
                    'non_standard_candidate': candidate,
                    'standard_candidates': standard_candidates_from_non
                })
            else:
                print(f"    → Maps to 관계 없음")
            
            print()
        
        # ===== 5단계: 모든 후보군에 대해 하이브리드 점수 기반 Re-ranking =====
        print(f"🎯 5단계: 모든 후보군에 대해 하이브리드 점수 기반 Re-ranking")
        print(f"{'='*60}")
        
        all_standard_candidates = []
        
        # 1. Standard 후보들에 대해 하이브리드 점수 사용
        print("  📊 Standard 후보들 하이브리드 점수:")
        for i, candidate in enumerate(standard_candidates, 1):
            source = candidate['_source']
            original_score = candidate['_score']
            
            # 하이브리드 후보에서 해당 후보 찾기
            hybrid_score = 0.0
            text_sim = 0.0
            semantic_sim = 0.0
            
            # candidates는 하이브리드 후보들이므로 매칭되는 것 찾기
            for hybrid_candidate in candidates:
                if hybrid_candidate['_source'].get('concept_id') == source.get('concept_id'):
                    hybrid_score = hybrid_candidate.get('hybrid_score', 0.0)
                    text_sim = hybrid_candidate.get('text_similarity', 0.0)
                    semantic_sim = hybrid_candidate.get('semantic_similarity', 0.0)
                    break
            
            print(f"    {i}. {source.get('concept_name', 'N/A')}")
            print(f"       하이브리드 점수: {hybrid_score:.3f} (텍스트: {text_sim:.3f}, 의미: {semantic_sim:.3f})")
            
            all_standard_candidates.append({
                'concept': source,
                'final_score': hybrid_score,  # 하이브리드 점수 사용
                'is_original_standard': True,
                'original_candidate': candidate,
                'elasticsearch_score': original_score,
                'hybrid_score': hybrid_score,
                'text_similarity': text_sim,
                'semantic_similarity': semantic_sim
            })
            print()
        
        # 2. Non-standard → Standard 후보들에 대해 Python 유사도 재계산
        print("  📊 Non-standard → Standard 후보들 Python 유사도 재계산:")
        for i, mapping in enumerate(non_standard_to_standard_mappings, 1):
            non_standard_source = mapping['non_standard_source']
            non_standard_candidate = mapping['non_standard_candidate']
            standard_candidates_list = mapping['standard_candidates']
            
            print(f"    Non-standard {i}: {non_standard_source.get('concept_name', 'N/A')}")
            
            for j, std_candidate in enumerate(standard_candidates_list, 1):
                # Python 유사도 재계산
                python_similarity = self.api._calculate_similarity(preprocessed_name, std_candidate.get('concept_name', ''))
                
                print(f"      Standard {j}: {std_candidate.get('concept_name', 'N/A')}")
                print(f"        Python 유사도: {python_similarity:.3f}")
                
                all_standard_candidates.append({
                    'concept': std_candidate,
                    'final_score': python_similarity,  # Python 유사도 사용
                    'is_original_standard': False,
                    'original_non_standard': non_standard_source,
                    'original_candidate': non_standard_candidate,
                    'python_similarity': python_similarity
                })
            
            print()
        
        # ===== 6단계: 점수 정규화 (0.0~1.0) → 최종 매핑 결과 =====
        print(f"📊 6단계: 점수 정규화 (0.0~1.0) → 최종 매핑 결과")
        print(f"{'='*60}")
        
        if not all_standard_candidates:
            print("❌ 처리된 후보 없음")
            return
        
        # 점수별 정렬 (하이브리드 점수 기준)
        sorted_candidates = sorted(all_standard_candidates, key=lambda x: x['final_score'], reverse=True)
        
        print("최종 후보 순위 (하이브리드 점수 기준):")
        for i, candidate in enumerate(sorted_candidates, 1):
            concept = candidate['concept']
            final_score = candidate['final_score']  # 하이브리드 점수
            is_standard = candidate['is_original_standard']
            mapping_type = "직접 Standard" if is_standard else "Non-standard → Standard"
            
            # 점수 정규화 (하이브리드 점수는 이미 0~1 사이)
            normalized_score = self.api._normalize_score(final_score)
            confidence = self.api._determine_confidence(normalized_score)
            
            print(f"  {i}. 하이브리드 점수: {final_score:.3f} → 정규화: {normalized_score:.3f} ({confidence})")
            
            # 세부 점수 정보
            if 'hybrid_score' in candidate:
                text_sim = candidate.get('text_similarity', 0.0)
                semantic_sim = candidate.get('semantic_similarity', 0.0)
                print(f"     └ 텍스트: {text_sim:.3f} | 의미: {semantic_sim:.3f}")
            
            print(f"     컨셉 ID: {concept.get('concept_id', 'N/A')}")
            print(f"     컨셉명: {concept.get('concept_name', 'N/A')}")
            print(f"     도메인: {concept.get('domain_id', 'N/A')}")
            print(f"     어휘체계: {concept.get('vocabulary_id', 'N/A')}")
            print(f"     매핑 방법: {mapping_type}")
            
            print()
        
        # 골든셋과 비교
        best_candidate = sorted_candidates[0]
        best_concept = best_candidate['concept']
        best_concept_id = str(best_concept.get('concept_id', ''))
        
        print(f"🎯 골든셋 비교:")
        if best_concept_id == golden_concept_id:
            print(f"  ✅ 성공! 골든셋과 정확히 일치")
        else:
            print(f"  ❌ 불일치")
        
        print(f"  예상: {golden_concept_id} - {golden_concept_name}")
        print(f"  실제: {best_concept_id} - {best_concept.get('concept_name', 'N/A')}")
        
        print(f"\n{'='*80}")


def main():
    """메인 테스트 함수"""
    print("🚀 엔티티 매핑 API 테스트 시작")
    
    # API 상태 확인
    tester = EntityMappingTester()
    health_check = tester.api.health_check()
    print(f"📊 API 상태: {health_check}")
    
    # 테스트 케이스 1: Adrenal Cushing's syndrome
    print("\n" + "="*80)
    print("📋 테스트 케이스 1: Adrenal Cushing's syndrome")
    print("="*80)
    tester.test_entity_mapping_6_steps(
        entity_name="Adrenal Cushing's syndrome",
        entity_type="condition",
        golden_concept_id="4030206",
        golden_concept_name="Adrenal Cushing's syndrome"
    )
    
    # 테스트 케이스 2: Acute Coronary Syndromes (ACS)
    print("\n" + "="*80)
    print("📋 테스트 케이스 2: Acute Coronary Syndromes (ACS)")
    print("="*80)
    tester.test_entity_mapping_6_steps(
        entity_name="Acute Coronary Syndromes (ACS)",
        entity_type="diagnostic",
        golden_concept_id="4215140",
        golden_concept_name="Acute coronary syndrome"
    )

    # 테스트 케이스 3: ST-segment elevation myocardial infarction (STEMI)
    print("\n" + "="*80)
    print("📋 테스트 케이스 3: ST-segment elevation myocardial infarction (STEMI)")
    print("="*80)
    tester.test_entity_mapping_6_steps(
        entity_name="ST-segment elevation myocardial infarction (STEMI)",
        entity_type="diagnostic",
        golden_concept_id="4296653",
        golden_concept_name="Acute ST segment elevation myocardial infarction"
    )
    
    print("\n🎉 모든 테스트 완료!")


if __name__ == "__main__":
    main()