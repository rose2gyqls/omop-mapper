#!/usr/bin/env python3
"""
3단계 엔티티 매핑 테스트 스크립트

1단계: Elasticsearch에서 concept_name 일치도로 10개 후보군 선별
2단계: 10개 후보군을 의미적 유사도 + Jaccard 유사도로 5개로 축소 + Non-std to Std 변환
3단계: 모든 Standard 후보군에 대해 의미적 유사도 + Jaccard 유사도로 최종 리랭킹

테스트 대상: entity_sample.xlsx의 10, 11, 12 시트
"""

import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'indexing'))

from omop_mapper.elasticsearch_client import ElasticsearchClient
from omop_mapper.entity_mapping_api import EntityMappingAPI
from sapbert_embedder import SapBERTEmbedder

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """NumPy 타입을 JSON 직렬화 가능한 타입으로 변환하는 인코더"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class ThreeStageMapper:
    """3단계 매핑 테스트 클래스"""
    
    def __init__(self, index_name: str = "concept", use_small_index: bool = False):
        """
        초기화
        
        Args:
            index_name: 사용할 인덱스 이름 (concept 또는 concept-small)
            use_small_index: concept-small 인덱스 사용 여부 (소문자 변환)
        """
        self.index_name = index_name
        self.use_small_index = use_small_index
        
        # Elasticsearch 클라이언트 초기화
        self.es_client = ElasticsearchClient()
        
        # EntityMappingAPI 초기화 (기존 로직 재사용)
        self.mapping_api = EntityMappingAPI()
        
        # SapBERT 임베딩 생성기 초기화 (임베딩 활성화)
        self.embedder = SapBERTEmbedder(enabled=True, batch_size=32)
        
        logger.info(f"3단계 매핑 테스트 초기화 완료 - 인덱스: {index_name}, 소문자 변환: {use_small_index}")
    
    def calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Jaccard 유사도 계산"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """의미적 유사도 계산 (SapBERT 임베딩 기반)"""
        try:
            if not self.embedder.enabled:
                return 0.0
            
            # 임베딩 생성
            embeddings = self.embedder.encode_texts([text1, text2], show_progress=False)
            
            if len(embeddings) != 2:
                return 0.0
            
            # 코사인 유사도 계산
            similarity = cosine_similarity(
                embeddings[0:1], 
                embeddings[1:2]
            )[0][0]
            
            # -1~1 범위를 0~1로 정규화
            return (similarity + 1.0) / 2.0
            
        except Exception as e:
            logger.warning(f"의미적 유사도 계산 실패: {e}")
            return 0.0
    
    def calculate_hybrid_score(self, text1: str, text2: str, 
                             jaccard_weight: float = 0.4, 
                             semantic_weight: float = 0.6) -> Dict[str, float]:
        """하이브리드 점수 계산 (Jaccard + 의미적 유사도)"""
        jaccard_sim = self.calculate_jaccard_similarity(text1, text2)
        semantic_sim = self.calculate_semantic_similarity(text1, text2)
        
        hybrid_score = jaccard_sim * jaccard_weight + semantic_sim * semantic_weight
        
        return {
            'hybrid_score': float(hybrid_score),
            'jaccard_similarity': float(jaccard_sim),
            'semantic_similarity': float(semantic_sim)
        }
    
    def stage1_elasticsearch_search(self, entity_name: str, domain_id: str = "Condition") -> List[Dict[str, Any]]:
        """1단계: Elasticsearch에서 10개 후보군 검색"""
        try:
            # concept-small 인덱스 사용 시 소문자 변환
            search_query = entity_name.lower() if self.use_small_index else entity_name
            
            # Elasticsearch 검색 실행
            results = self.es_client.search_concepts(
                query=search_query,
                domain_ids=[domain_id] if domain_id else None,
                standard_concept_only=False,  # Standard/Non-standard 모두 포함
                limit=10
            )
            
            # 결과를 딕셔너리 형태로 변환
            candidates = []
            for result in results:
                candidates.append({
                    '_source': {
                        'concept_id': result.concept_id,
                        'concept_name': result.concept_name,
                        'domain_id': result.domain_id,
                        'vocabulary_id': result.vocabulary_id,
                        'concept_class_id': result.concept_class_id,
                        'standard_concept': result.standard_concept,
                        'concept_code': result.concept_code
                    },
                    '_score': result.score
                })
            
            logger.info(f"1단계 완료: {len(candidates)}개 후보군 검색 ('{entity_name}' -> '{search_query}')")
            return candidates
            
        except Exception as e:
            logger.error(f"1단계 Elasticsearch 검색 실패: {e}")
            return []
    
    def stage2_hybrid_filtering_and_nonstd_to_std(self, entity_name: str, candidates: List[Dict[str, Any]], 
                                                 domain_id: str = "Condition") -> List[Dict[str, Any]]:
        """2단계: 하이브리드 점수로 5개 선별 + Non-std to Std 변환"""
        if not candidates:
            return []
        
        # 각 후보에 대해 하이브리드 점수 계산
        scored_candidates = []
        for candidate in candidates:
            source = candidate['_source']
            concept_name = source.get('concept_name', '')
            
            # 하이브리드 점수 계산
            scores = self.calculate_hybrid_score(entity_name, concept_name)
            
            scored_candidates.append({
                'candidate': candidate,
                'hybrid_score': float(scores['hybrid_score']),
                'jaccard_similarity': float(scores['jaccard_similarity']),
                'semantic_similarity': float(scores['semantic_similarity']),
                'elasticsearch_score': float(candidate['_score'])
            })
        
        # 하이브리드 점수 기준 정렬 후 상위 5개 선택
        scored_candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
        top5_candidates = scored_candidates[:5]
        
        # Standard/Non-standard 분류 및 Non-std to Std 변환
        all_standard_candidates = []
        
        for scored_candidate in top5_candidates:
            candidate = scored_candidate['candidate']
            source = candidate['_source']
            
            if source.get('standard_concept') == 'S':
                # Standard 개념: 직접 추가
                all_standard_candidates.append({
                    'concept': source,
                    'is_original_standard': True,
                    'original_candidate': candidate,
                    'stage2_hybrid_score': float(scored_candidate['hybrid_score']),
                    'stage2_jaccard_similarity': float(scored_candidate['jaccard_similarity']),
                    'stage2_semantic_similarity': float(scored_candidate['semantic_similarity']),
                    'elasticsearch_score': float(scored_candidate['elasticsearch_score'])
                })
            else:
                # Non-standard 개념: Standard 후보들 조회
                concept_id = str(source.get('concept_id', ''))
                try:
                    standard_candidates = self.mapping_api._get_standard_candidates(concept_id, domain_id)
                    
                    for std_candidate in standard_candidates:
                        all_standard_candidates.append({
                            'concept': std_candidate,
                            'is_original_standard': False,
                            'original_non_standard': source,
                            'original_candidate': candidate,
                            'stage2_hybrid_score': float(scored_candidate['hybrid_score']),
                            'stage2_jaccard_similarity': float(scored_candidate['jaccard_similarity']),
                            'stage2_semantic_similarity': float(scored_candidate['semantic_similarity']),
                            'elasticsearch_score': 0.0  # Non-std -> Std는 ES 점수 없음
                        })
                        
                except Exception as e:
                    logger.warning(f"Non-std to Std 변환 실패 (concept_id: {concept_id}): {e}")
        
        logger.info(f"2단계 완료: {len(top5_candidates)}개 선별 -> {len(all_standard_candidates)}개 Standard 후보군")
        return all_standard_candidates
    
    def stage3_final_reranking(self, entity_name: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """3단계: 모든 Standard 후보군에 대해 최종 리랭킹"""
        if not candidates:
            return []
        
        # 각 Standard 후보에 대해 최종 하이브리드 점수 계산
        final_scored_candidates = []
        
        for candidate in candidates:
            concept = candidate['concept']
            concept_name = concept.get('concept_name', '')
            
            # 최종 하이브리드 점수 계산
            scores = self.calculate_hybrid_score(entity_name, concept_name)
            
            final_scored_candidates.append({
                **candidate,  # 기존 정보 유지
                'stage3_hybrid_score': float(scores['hybrid_score']),
                'stage3_jaccard_similarity': float(scores['jaccard_similarity']),
                'stage3_semantic_similarity': float(scores['semantic_similarity']),
                'final_score': float(scores['hybrid_score'])  # 최종 점수
            })
        
        # 최종 점수 기준 정렬
        final_scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        logger.info(f"3단계 완료: {len(final_scored_candidates)}개 후보군 최종 리랭킹")
        return final_scored_candidates
    
    def map_entity_3stage(self, entity_name: str, domain_id: str = "Condition") -> Dict[str, Any]:
        """3단계 매핑 전체 프로세스"""
        result = {
            'entity_name': entity_name,
            'domain_id': domain_id,
            'stage1_candidates': [],
            'stage2_candidates': [],
            'stage3_candidates': [],
            'final_mapping': None
        }
        
        try:
            # 1단계: Elasticsearch 검색
            stage1_candidates = self.stage1_elasticsearch_search(entity_name, domain_id)
            result['stage1_candidates'] = [
                {
                    'concept_id': c['_source'].get('concept_id'),
                    'concept_name': c['_source'].get('concept_name'),
                    'standard_concept': c['_source'].get('standard_concept'),
                    'elasticsearch_score': float(c['_score'])
                }
                for c in stage1_candidates
            ]
            
            # 2단계: 하이브리드 필터링 + Non-std to Std
            stage2_candidates = self.stage2_hybrid_filtering_and_nonstd_to_std(
                entity_name, stage1_candidates, domain_id
            )
            result['stage2_candidates'] = [
                {
                    'concept_id': c['concept'].get('concept_id'),
                    'concept_name': c['concept'].get('concept_name'),
                    'is_original_standard': c['is_original_standard'],
                    'stage2_hybrid_score': float(c['stage2_hybrid_score']),
                    'stage2_jaccard_similarity': float(c['stage2_jaccard_similarity']),
                    'stage2_semantic_similarity': float(c['stage2_semantic_similarity']),
                    'elasticsearch_score': float(c['elasticsearch_score'])
                }
                for c in stage2_candidates
            ]
            
            # 3단계: 최종 리랭킹
            stage3_candidates = self.stage3_final_reranking(entity_name, stage2_candidates)
            result['stage3_candidates'] = [
                {
                    'concept_id': c['concept'].get('concept_id'),
                    'concept_name': c['concept'].get('concept_name'),
                    'is_original_standard': c['is_original_standard'],
                    'stage3_hybrid_score': float(c['stage3_hybrid_score']),
                    'stage3_jaccard_similarity': float(c['stage3_jaccard_similarity']),
                    'stage3_semantic_similarity': float(c['stage3_semantic_similarity']),
                    'final_score': float(c['final_score'])
                }
                for c in stage3_candidates
            ]
            
            # 최종 매핑 결과
            if stage3_candidates:
                best_candidate = stage3_candidates[0]
                result['final_mapping'] = {
                    'concept_id': best_candidate['concept'].get('concept_id'),
                    'concept_name': best_candidate['concept'].get('concept_name'),
                    'domain_id': best_candidate['concept'].get('domain_id'),
                    'vocabulary_id': best_candidate['concept'].get('vocabulary_id'),
                    'final_score': float(best_candidate['final_score']),
                    'mapping_method': 'direct_standard' if best_candidate['is_original_standard'] else 'non_standard_to_standard'
                }
            
        except Exception as e:
            logger.error(f"3단계 매핑 실패 ({entity_name}): {e}")
            result['error'] = str(e)
        
        return result


def load_test_entities(excel_path: str, sheets: List[str]) -> List[Dict[str, Any]]:
    """엑셀 파일에서 테스트 엔티티 로드"""
    entities = []
    
    for sheet_name in sheets:
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            # NaN 값 제거
            valid_entities = df[df['entity_plain_name'].notna()]['entity_plain_name'].unique()
            
            for entity_name in valid_entities:
                entities.append({
                    'page': sheet_name,
                    'entity_plain_name': str(entity_name).strip()
                })
                
        except Exception as e:
            logger.error(f"시트 '{sheet_name}' 로드 실패: {e}")
    
    logger.info(f"테스트 엔티티 로드 완료: {len(entities)}개")
    return entities


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='3단계 엔티티 매핑 테스트')
    parser.add_argument('--index', choices=['concept', 'concept-small'], 
                       default='concept', help='사용할 인덱스 (기본값: concept)')
    parser.add_argument('--output', default='3stage_mapping_results.json', 
                       help='결과 저장 파일명 (기본값: 3stage_mapping_results.json)')
    parser.add_argument('--limit', type=int, default=None, 
                       help='테스트할 엔티티 수 제한 (기본값: 전체)')
    
    args = parser.parse_args()
    
    # 인덱스 설정
    use_small_index = (args.index == 'concept-small')
    
    print("=" * 80)
    print("🔬 3단계 엔티티 매핑 테스트")
    print("=" * 80)
    print(f"📊 인덱스: {args.index}")
    print(f"📄 출력 파일: {args.output}")
    print(f"🔢 제한: {args.limit if args.limit else '없음'}")
    print("=" * 80)
    
    # 매핑 테스터 초기화
    mapper = ThreeStageMapper(index_name=args.index, use_small_index=use_small_index)
    
    # 테스트 엔티티 로드
    excel_path = "/home/work/skku/hyo/omop-mapper/data/entity_sample.xlsx"
    test_sheets = ['10', '11', '12']
    
    entities = load_test_entities(excel_path, test_sheets)
    
    if args.limit:
        entities = entities[:args.limit]
    
    print(f"📝 테스트 대상: {len(entities)}개 엔티티")
    print()
    
    # 매핑 테스트 실행
    results = []
    
    for i, entity_info in enumerate(entities, 1):
        entity_name = entity_info['entity_plain_name']
        page = entity_info['page']
        
        print(f"[{i}/{len(entities)}] 테스트 중: '{entity_name}' (페이지: {page})")
        
        # 3단계 매핑 실행
        mapping_result = mapper.map_entity_3stage(entity_name)
        mapping_result['page'] = page
        
        results.append(mapping_result)
        
        # 간단한 결과 출력
        if mapping_result.get('final_mapping'):
            final = mapping_result['final_mapping']
            print(f"  ✅ 매핑 성공: {final['concept_name']} (점수: {final['final_score']:.3f})")
        else:
            print(f"  ❌ 매핑 실패")
        print()
    
    # 결과 저장 (NumPy 타입 변환 포함)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    print("=" * 80)
    print("🏁 테스트 완료")
    print("=" * 80)
    print(f"📄 결과 저장됨: {args.output}")
    print(f"📊 총 {len(results)}개 엔티티 테스트 완료")
    
    # 성공률 계산
    success_count = sum(1 for r in results if r.get('final_mapping'))
    success_rate = success_count / len(results) * 100 if results else 0
    print(f"✅ 매핑 성공률: {success_count}/{len(results)} ({success_rate:.1f}%)")


if __name__ == "__main__":
    main()
