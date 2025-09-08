#!/usr/bin/env python3
"""
CONCEPT 데이터 인덱싱 실행 스크립트

사용법:
    # 전체 데이터 인덱싱
    python run_concept_indexing.py

    # 테스트 실행
    python run_concept_indexing.py --test

    # 특정 개수만 인덱싱
    python run_concept_indexing.py --max-concepts 10000

    # 일부 건너뛰고 인덱싱
    python run_concept_indexing.py --skip-concepts 5000 --max-concepts 10000
"""

import argparse
import sys
import os
from pathlib import Path

# 현재 스크립트 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
indexing_dir = current_dir / "indexing"
sys.path.insert(0, str(indexing_dir))

from indexing.concept_indexer_with_sapbert import ConceptIndexerWithSapBERT


def main():
    parser = argparse.ArgumentParser(description="CONCEPT 데이터 Elasticsearch 인덱싱")
    
    # 기본 설정
    parser.add_argument("--csv-path", 
                       default="/home/work/skku/hyo/omop-mapper/data/CONCEPT.csv",
                       help="CONCEPT.csv 파일 경로")
    parser.add_argument("--es-host", default="3.35.110.161", help="Elasticsearch 호스트")
    parser.add_argument("--es-port", type=int, default=9200, help="Elasticsearch 포트")
    parser.add_argument("--es-username", default="elastic", help="Elasticsearch 사용자명")
    parser.add_argument("--es-password", default="snomed", help="Elasticsearch 비밀번호")
    parser.add_argument("--index-name", default="concepts", help="인덱스명")
    
    # 모델 설정
    parser.add_argument("--model-name", 
                       default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                       help="SapBERT 모델명")
    parser.add_argument("--gpu-device", type=int, default=0, help="GPU 디바이스 번호")
    
    # 처리 설정
    parser.add_argument("--batch-size", type=int, default=128, help="임베딩 배치 크기")
    parser.add_argument("--chunk-size", type=int, default=1000, help="데이터 처리 청크 크기")
    
    # 데이터 범위 설정
    parser.add_argument("--max-concepts", type=int, default=None, 
                       help="최대 처리할 concept 수 (기본: 전체)")
    parser.add_argument("--skip-concepts", type=int, default=0, 
                       help="건너뛸 concept 수")
    
    # 기타 옵션
    parser.add_argument("--delete-existing", action="store_true", 
                       help="기존 인덱스 삭제")
    parser.add_argument("--test", action="store_true", 
                       help="테스트 모드 (1000개만 처리)")
    parser.add_argument("--no-search-test", action="store_true",
                       help="검색 테스트 건너뛰기")
    
    args = parser.parse_args()
    
    # 테스트 모드 설정
    if args.test:
        args.max_concepts = 1000
        args.batch_size = 64
        args.chunk_size = 100
        args.index_name = "test_concepts"
        args.delete_existing = True
        print("🧪 테스트 모드: 1000개 concept만 처리합니다.")
    
    # 설정 출력
    print("=== CONCEPT 인덱싱 설정 ===")
    print(f"CSV 파일: {args.csv_path}")
    print(f"Elasticsearch: {args.es_host}:{args.es_port}")
    print(f"ES 사용자: {args.es_username}")
    print(f"인덱스명: {args.index_name}")
    print(f"모델: {args.model_name}")
    print(f"GPU 디바이스: {args.gpu_device}")
    print(f"배치 크기: {args.batch_size}")
    print(f"청크 크기: {args.chunk_size}")
    print(f"최대 처리 수: {args.max_concepts if args.max_concepts else '전체'}")
    print(f"건너뛸 수: {args.skip_concepts}")
    print(f"기존 인덱스 삭제: {args.delete_existing}")
    print("="*30)
    
    try:
        # 인덱서 초기화
        indexer = ConceptIndexerWithSapBERT(
            csv_file_path=args.csv_path,
            es_host=args.es_host,
            es_port=args.es_port,
            es_username=args.es_username,
            es_password=args.es_password,
            index_name=args.index_name,
            model_name=args.model_name,
            gpu_device=args.gpu_device,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size
        )
        
        # 인덱싱 실행
        success = indexer.run_full_indexing(
            delete_existing_index=args.delete_existing,
            max_concepts=args.max_concepts,
            skip_concepts=args.skip_concepts
        )
        
        if success:
            print("✅ 인덱싱이 성공적으로 완료되었습니다!")
            
            # 검색 테스트 (옵션)
            if not args.no_search_test:
                print("\n🔍 검색 테스트를 실행합니다...")
                indexer.test_search()
        else:
            print("❌ 인덱싱에 실패했습니다.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # 리소스 정리
        if 'indexer' in locals():
            indexer.cleanup()


if __name__ == "__main__":
    main()
