#!/usr/bin/env python3
"""
OMOP CDM 인덱싱 스크립트

설정값을 수정하고 실행하거나, CLI 옵션으로 오버라이드할 수 있습니다.

Usage:
    # 기본 설정으로 실행
    python run_indexing.py
    
    # CLI 옵션으로 실행
    python run_indexing.py local_csv --data-folder /path/to/data --tables concept-small synonym
    python run_indexing.py postgres --tables concept-small relationship synonym
    
    # 테스트 (일부 데이터만)
    python run_indexing.py local_csv --max-rows 10000
"""

import sys
import argparse
import logging
import time
from pathlib import Path

# ============================================================================
# 기본 설정 (CLI 옵션이 없으면 이 값 사용)
# ============================================================================

# 데이터 소스 타입: 'local_csv' 또는 'postgres'
DEFAULT_SOURCE = 'local_csv'

# 인덱싱할 테이블 목록
# 옵션: 'concept-small', 'synonym', 'relationship', 'concept'
DEFAULT_TABLES = ['concept-small', 'synonym', 'relationship']

# ----------------------------------------------------------------------------
# Local CSV 설정
# ----------------------------------------------------------------------------
DEFAULT_DATA_FOLDER = '/workspace/omop-mapper/data/omop-cdm'

# ----------------------------------------------------------------------------
# PostgreSQL 설정
# ----------------------------------------------------------------------------
DEFAULT_PG_HOST = '172.23.100.146'
DEFAULT_PG_PORT = '1341'
DEFAULT_PG_DBNAME = 'cdm_public'
DEFAULT_PG_USER = 'cdmreader'
DEFAULT_PG_PASSWORD = 'scdm2025!@'

# ----------------------------------------------------------------------------
# Elasticsearch 설정
# ----------------------------------------------------------------------------
DEFAULT_ES_HOST = '3.35.110.161'
DEFAULT_ES_PORT = 9200
DEFAULT_ES_USER = 'elastic'
DEFAULT_ES_PASSWORD = 'snomed'

# ----------------------------------------------------------------------------
# 인덱싱 옵션 (대용량/GPU 최적화 기본값)
# ----------------------------------------------------------------------------
DEFAULT_GPU = 0                 # GPU 번호 (-1: CPU 사용)
DEFAULT_EMBEDDINGS = True       # SapBERT 임베딩 포함 여부
DEFAULT_LOWERCASE = True        # concept_name 소문자 변환 여부
DEFAULT_BATCH_SIZE = 512        # 임베딩 배치 크기 (GPU: 512~1024 권장)
DEFAULT_CHUNK_SIZE = 10000      # 데이터 청크 크기 (클수록 GPU 활용·ES round-trip 감소)

# ============================================================================
# 메인 코드
# ============================================================================

def setup_logging(source_type: str) -> str:
    """로깅 설정"""
    log_file = f'indexing_{source_type}_{time.strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


def parse_args():
    """CLI 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='OMOP CDM Elasticsearch 인덱싱',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 데이터 소스 (선택적)
    parser.add_argument(
        'source_type', nargs='?',
        choices=['local_csv', 'postgres'],
        default=DEFAULT_SOURCE,
        help=f'데이터 소스 타입 (기본: {DEFAULT_SOURCE})'
    )
    
    # 공통 옵션
    parser.add_argument('--tables', nargs='+',
        choices=['concept', 'concept-small', 'relationship', 'synonym'],
        default=DEFAULT_TABLES,
        help=f'인덱싱할 테이블 (기본: {DEFAULT_TABLES})')
    parser.add_argument('--gpu', type=int, default=DEFAULT_GPU,
        help=f'GPU 번호, -1은 CPU (기본: {DEFAULT_GPU})')
    parser.add_argument('--no-embeddings', action='store_true',
        help='SapBERT 임베딩 비활성화')
    parser.add_argument('--no-lowercase', action='store_true',
        help='소문자 변환 비활성화')
    parser.add_argument('--max-rows', type=int, default=None,
        help='최대 처리 행 수 (테스트용)')
    parser.add_argument('--resume', action='store_true',
        help='끊긴 부분부터 재개: 기존 인덱스 유지 후, 이미 인덱싱된 행 수만큼 건너뛰고 이어서 인덱싱')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
        help=f'임베딩 배치 크기 (기본: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
        help=f'데이터 청크 크기 (기본: {DEFAULT_CHUNK_SIZE})')
    
    # Elasticsearch 옵션
    parser.add_argument('--es-host', default=DEFAULT_ES_HOST,
        help=f'Elasticsearch 호스트 (기본: {DEFAULT_ES_HOST})')
    parser.add_argument('--es-port', type=int, default=DEFAULT_ES_PORT,
        help=f'Elasticsearch 포트 (기본: {DEFAULT_ES_PORT})')
    parser.add_argument('--es-user', default=DEFAULT_ES_USER,
        help=f'Elasticsearch 사용자 (기본: {DEFAULT_ES_USER})')
    parser.add_argument('--es-password', default=DEFAULT_ES_PASSWORD,
        help='Elasticsearch 비밀번호')
    
    # Local CSV 옵션
    parser.add_argument('--data-folder', default=DEFAULT_DATA_FOLDER,
        help=f'CSV 데이터 폴더 (기본: {DEFAULT_DATA_FOLDER})')
    
    # PostgreSQL 옵션
    parser.add_argument('--pg-host', default=DEFAULT_PG_HOST,
        help=f'PostgreSQL 호스트 (기본: {DEFAULT_PG_HOST})')
    parser.add_argument('--pg-port', default=DEFAULT_PG_PORT,
        help=f'PostgreSQL 포트 (기본: {DEFAULT_PG_PORT})')
    parser.add_argument('--pg-dbname', default=DEFAULT_PG_DBNAME,
        help=f'PostgreSQL DB명 (기본: {DEFAULT_PG_DBNAME})')
    parser.add_argument('--pg-user', default=DEFAULT_PG_USER,
        help=f'PostgreSQL 사용자 (기본: {DEFAULT_PG_USER})')
    parser.add_argument('--pg-password', default=DEFAULT_PG_PASSWORD,
        help='PostgreSQL 비밀번호')
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()
    
    # 경로 설정
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent / "indexing"))
    
    # 로깅 설정
    log_file = setup_logging(args.source_type)
    logger = logging.getLogger(__name__)
    
    print("=" * 70)
    print("OMOP CDM Elasticsearch 인덱싱")
    print("=" * 70)
    print(f"데이터 소스: {args.source_type}")
    print(f"테이블: {args.tables}")
    print(f"Elasticsearch: {args.es_host}:{args.es_port}")
    print(f"GPU: {args.gpu}")
    print(f"임베딩: {'비활성화' if args.no_embeddings else '활성화'}")
    print(f"로그: {log_file}")
    print("=" * 70)
    
    try:
        from indexing.unified_indexer import UnifiedIndexer, create_data_source
        
        # 1. 데이터 소스 생성
        if args.source_type == 'local_csv':
            print(f"\n데이터 폴더: {args.data_folder}")
            
            # concept-small 전처리
            if 'concept-small' in args.tables:
                print("\n[1/2] CONCEPT_SMALL.csv 확인 중...")
                from prepare_concept_small import create_concept_small
                
                concept_small_path = Path(args.data_folder) / 'CONCEPT_SMALL.csv'
                if not concept_small_path.exists():
                    print("  -> 생성 중...")
                    create_concept_small(args.data_folder)
                else:
                    print("  -> 이미 존재 (스킵)")
            
            data_source = create_data_source(
                'local_csv',
                data_folder=args.data_folder
            )
            
        elif args.source_type == 'postgres':
            print(f"\nPostgreSQL: {args.pg_host}:{args.pg_port}/{args.pg_dbname}")
            print("\n[1/2] PostgreSQL 연결 중...")
            
            data_source = create_data_source(
                'postgres',
                host=args.pg_host,
                port=args.pg_port,
                dbname=args.pg_dbname,
                user=args.pg_user,
                password=args.pg_password
            )
            print("  -> 연결 성공")
        
        # 2. 인덱서 생성 및 실행
        print("\n[2/2] Elasticsearch 인덱싱...")
        
        indexer = UnifiedIndexer(
            data_source=data_source,
            es_host=args.es_host,
            es_port=args.es_port,
            es_username=args.es_user,
            es_password=args.es_password,
            gpu_device=args.gpu,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            include_embeddings=not args.no_embeddings,
            lowercase=not args.no_lowercase
        )
        
        results = indexer.index_all(
            delete_existing=not args.resume,
            max_rows=args.max_rows,
            tables=args.tables
        )
        
        # 3. 결과 출력
        print("\n" + "=" * 70)
        print("결과:")
        for table, success in results.items():
            status = "✓ 성공" if success else "✗ 실패"
            print(f"  {table}: {status}")
        print("=" * 70)
        
        indexer.cleanup()
        
        if all(results.values()):
            print("\n완료!")
            return 0
        else:
            print("\n일부 실패. 로그 확인 필요.")
            return 1
            
    except ImportError as e:
        print(f"\n오류: {e}")
        print("pip install -r requirements.txt 실행 필요")
        return 1
    except FileNotFoundError as e:
        print(f"\n파일 없음: {e}")
        return 1
    except ConnectionError as e:
        print(f"\n연결 실패: {e}")
        return 1
    except Exception as e:
        logger.error(f"오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\n오류: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
