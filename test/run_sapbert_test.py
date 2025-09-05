#!/usr/bin/env python3
"""
SapBERT 임베딩 테스트 실행 스크립트
간단하게 SapBERT 성능을 테스트할 수 있는 스크립트입니다.
"""

import argparse
import os
import sys

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sapbert_embedding_test import SapBERTEmbeddingTester


def quick_test(csv_path, sample_size=100):
    """
    빠른 SapBERT 테스트 실행
    
    Args:
        csv_path: CONCEPT.csv 파일 경로
        sample_size: 테스트할 샘플 크기
    """
    print("🚀 SapBERT 빠른 테스트 시작")
    print(f"샘플 크기: {sample_size}")
    print("-" * 50)
    
    # 테스터 초기화
    tester = SapBERTEmbeddingTester()
    
    # 데이터 로드
    df = tester.load_concept_data(csv_path, sample_size=sample_size)
    
    # 임베딩 생성
    concept_names = df['concept_name'].tolist()
    embeddings = tester.get_embeddings(concept_names, batch_size=16)
    
    # 유사성 분석
    similarity_results = tester.analyze_similarity(df, embeddings)
    
    # 결과 요약
    print("\n" + "="*50)
    print("📊 테스트 결과 요약")
    print("="*50)
    print(f"분석된 concept 수: {len(df):,}")
    print(f"임베딩 차원: {embeddings.shape[1]}")
    print(f"전체 평균 유사성: {similarity_results['overall_avg_similarity']:.4f}")
    
    print("\n도메인별 평균 유사성:")
    for domain, sim in sorted(similarity_results['domain_similarities'].items(), 
                             key=lambda x: x[1], reverse=True):
        print(f"  {domain}: {sim:.4f}")
    
    # 성능 평가
    avg_sim = similarity_results['overall_avg_similarity']
    print(f"\n💡 성능 평가:")
    if avg_sim > 0.7:
        print("   🟢 우수: 임베딩 품질이 매우 좋습니다!")
    elif avg_sim > 0.5:
        print("   🟡 양호: 임베딩 품질이 괜찮습니다.")
    else:
        print("   🔴 개선 필요: 임베딩 품질 향상이 필요합니다.")
    
    # 유사성 예시
    print(f"\n🔍 유사성 분석 예시:")
    for i, example in enumerate(similarity_results['sample_examples'][:2]):
        print(f"\n{i+1}. '{example['original']['name']}' ({example['original']['domain']})")
        print("   유사한 엔티티들:")
        for j, similar in enumerate(example['similar'][:3]):
            print(f"   {j+1}. {similar['name']} - 유사도: {similar['similarity']:.4f}")
    
    print("\n🎉 빠른 테스트 완료!")
    return tester, df, embeddings, similarity_results


def comprehensive_test(csv_path, sample_size=1000, output_dir="./sapbert_results"):
    """
    종합적인 SapBERT 테스트 실행 (시각화 포함)
    
    Args:
        csv_path: CONCEPT.csv 파일 경로  
        sample_size: 테스트할 샘플 크기
        output_dir: 결과 저장 디렉토리
    """
    print("🎯 SapBERT 종합 테스트 시작")
    print(f"샘플 크기: {sample_size}")
    print(f"결과 저장 디렉토리: {output_dir}")
    print("-" * 50)
    
    tester = SapBERTEmbeddingTester()
    results = tester.run_comprehensive_test(
        csv_path=csv_path,
        sample_size=sample_size,
        output_dir=output_dir
    )
    
    return results


def test_specific_terms():
    """
    특정 의료 용어들로 SapBERT 테스트
    """
    print("🔬 특정 의료 용어 테스트 시작")
    print("-" * 50)
    
    # 테스트할 의료 용어들
    medical_terms = [
        "diabetes mellitus",
        "hypertension", 
        "myocardial infarction",
        "pneumonia",
        "influenza",
        "covid-19",
        "headache",
        "fever",
        "cough",
        "chest pain"
    ]
    
    tester = SapBERTEmbeddingTester()
    
    print(f"테스트 용어들: {', '.join(medical_terms)}")
    
    # 임베딩 생성
    embeddings = tester.get_embeddings(medical_terms, batch_size=8)
    
    # 유사성 계산
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(embeddings)
    
    # 가장 유사한 쌍들 찾기
    pairs = []
    for i in range(len(medical_terms)):
        for j in range(i+1, len(medical_terms)):
            pairs.append((medical_terms[i], medical_terms[j], similarity_matrix[i, j]))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n📈 가장 유사한 용어 쌍들 (상위 5개):")
    for i, (term1, term2, sim) in enumerate(pairs[:5]):
        print(f"{i+1}. {term1} ↔ {term2}: {sim:.4f}")
    
    print(f"\n📉 가장 다른 용어 쌍들 (하위 3개):")
    for i, (term1, term2, sim) in enumerate(pairs[-3:]):
        print(f"{i+1}. {term1} ↔ {term2}: {sim:.4f}")
    
    print("\n🎉 특정 용어 테스트 완료!")
    return medical_terms, embeddings, similarity_matrix


def main():
    parser = argparse.ArgumentParser(description='SapBERT 의료 엔티티 임베딩 테스트')
    parser.add_argument('--csv_path', default='../data/CONCEPT.csv', 
                       help='CONCEPT.csv 파일 경로')
    parser.add_argument('--mode', choices=['quick', 'comprehensive', 'terms'], 
                       default='quick', help='테스트 모드')
    parser.add_argument('--sample_size', type=int, default=100, 
                       help='샘플 크기 (quick: 100, comprehensive: 1000 권장)')
    parser.add_argument('--output_dir', default='./sapbert_results',
                       help='결과 저장 디렉토리 (comprehensive 모드에서만 사용)')
    
    args = parser.parse_args()
    
    print("🧬 SapBERT 의료 엔티티 임베딩 테스트")
    print("=" * 60)
    
    try:
        if args.mode == 'quick':
            quick_test(args.csv_path, args.sample_size)
            
        elif args.mode == 'comprehensive':
            comprehensive_test(args.csv_path, args.sample_size, args.output_dir)
            
        elif args.mode == 'terms':
            test_specific_terms()
            
    except FileNotFoundError:
        print(f"❌ 오류: {args.csv_path} 파일을 찾을 수 없습니다.")
        print("올바른 CONCEPT.csv 파일 경로를 확인해주세요.")
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        print("문제가 지속되면 로그를 확인해주세요.")


if __name__ == "__main__":
    main()
