#!/usr/bin/env python3
"""
Elasticsearch 인덱스 조사 실행 스크립트

사용법:
    python run_index_inspection.py                    # 기본 설정으로 실행
    python run_index_inspection.py --host 127.0.0.1  # 특정 호스트 지정
    python run_index_inspection.py --help             # 도움말 보기
"""

import argparse
import sys
from elasticsearch_index_inspector import ElasticsearchIndexInspector


def parse_arguments():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="Elasticsearch 인덱스 내용 조사 및 로그 저장",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  %(prog)s                                    # 기본 설정으로 실행
  %(prog)s --host 127.0.0.1 --port 9200      # 로컬 ES 서버 조사
  %(prog)s --host 3.35.110.161 --username elastic --password snomed
        """
    )
    
    parser.add_argument(
        '--host',
        type=str,
        help='Elasticsearch 서버 호스트 (기본값: 환경변수 ES_SERVER_HOST 또는 3.35.110.161)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        help='Elasticsearch 서버 포트 (기본값: 환경변수 ES_SERVER_PORT 또는 9200)'
    )
    
    parser.add_argument(
        '--username',
        type=str,
        help='사용자명 (기본값: 환경변수 ES_SERVER_USERNAME 또는 elastic)'
    )
    
    parser.add_argument(
        '--password',
        type=str,
        help='비밀번호 (기본값: 환경변수 ES_SERVER_PASSWORD 또는 snomed)'
    )
    
    parser.add_argument(
        '--scheme',
        type=str,
        choices=['http', 'https'],
        default='http',
        help='연결 스키마 (기본값: http)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=5,
        help='각 인덱스에서 조회할 샘플 문서 수 (기본값: 5)'
    )
    
    parser.add_argument(
        '--indices',
        nargs='+',
        help='특정 인덱스만 조사 (기본값: 모든 인덱스)'
    )
    
    return parser.parse_args()


def main():
    """메인 실행 함수"""
    args = parse_arguments()
    
    try:
        print("🔍 Elasticsearch 인덱스 조사를 시작합니다...")
        print(f"서버: {args.host or '환경변수/기본값'}:{args.port or '환경변수/기본값'}")
        print(f"인증: {args.username or '환경변수/기본값'}")
        print("-" * 50)
        
        # 인덱스 검사기 생성
        inspector = ElasticsearchIndexInspector(
            host=args.host,
            port=args.port,
            username=args.username,
            password=args.password,
            scheme=args.scheme
        )
        
        if args.indices:
            # 특정 인덱스만 조사
            print(f"📋 지정된 인덱스 조사: {', '.join(args.indices)}")
            for index_name in args.indices:
                try:
                    inspector.inspect_index(index_name)
                except Exception as e:
                    print(f"❌ 인덱스 {index_name} 조사 실패: {e}")
        else:
            # 모든 인덱스 조사
            inspector.inspect_all_indices()
        
        print(f"\n✅ 인덱스 조사가 완료되었습니다!")
        print(f"📄 로그 파일: {inspector.log_filename}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
