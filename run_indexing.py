#!/usr/bin/env python3
"""
CONCEPT 인덱스 생성 실행 스크립트

사용법:
    python run_indexing.py                    # concepts-small 인덱스 생성 (소문자 변환, GPU 0)
    python run_indexing.py --std              # concepts 인덱스 생성 (원본 유지, GPU 0)
    python run_indexing.py --gpu 1            # concepts-small 인덱스 생성 (GPU 1)
    python run_indexing.py --std --gpu 0      # concepts 인덱스 생성 (GPU 0)
    python run_indexing.py --resume           # 중단된 지점부터 재시작 (concepts-small)
    python run_indexing.py --std --resume     # 중단된 지점부터 재시작 (concepts)

기능:
    - concepts-small: concept_name을 모두 소문자로 변환하여 저장
    - concepts: 원본 concept_name 그대로 저장
    - SapBERT 임베딩 포함
    - GPU 선택 가능
    - Resume 기능: 중단된 지점부터 인덱싱 재시작
"""

import sys
import os
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# indexing 모듈 경로 추가
indexing_dir = current_dir / "indexing"
sys.path.insert(0, str(indexing_dir))

# 인덱서 임포트 및 실행
try:
    from indexing.concept_indexer_with_sapbert import main
    
    if __name__ == "__main__":
        import argparse
        
        # 명령행 인수 파싱
        parser = argparse.ArgumentParser(description='CONCEPT 인덱스 생성')
        parser.add_argument('--std', action='store_true', help='concepts 인덱스 생성 (원본 유지)')
        parser.add_argument('--gpu', type=int, default=0, help='사용할 GPU 번호 (기본값: 0)')
        parser.add_argument('--resume', action='store_true', help='중단된 지점부터 인덱싱 재시작')
        
        args = parser.parse_args()
        
        # 인덱스 타입 결정
        if args.std:
            create_small = False
            index_type = f"concepts (원본 유지, GPU {args.gpu})"
        else:
            create_small = True
            index_type = f"concepts-small (소문자 변환, GPU {args.gpu})"
        
        print("=" * 60)
        print(f"{index_type} 인덱스 생성 시작")
        print("=" * 60)
        
        # 메인 함수 실행
        main(create_small_index=create_small, gpu_device=args.gpu, resume=args.resume)
        
        print("=" * 60)
        print(f"{index_type} 인덱스 생성 완료")
        print("=" * 60)
        
except ImportError as e:
    print(f"모듈 임포트 오류: {e}")
    print("indexing 디렉토리의 모든 필요한 파일이 있는지 확인해주세요.")
    sys.exit(1)
except Exception as e:
    print(f"실행 중 오류 발생: {e}")
    sys.exit(1)
