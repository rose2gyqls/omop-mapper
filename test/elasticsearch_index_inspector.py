#!/usr/bin/env python3
"""
Elasticsearch 인덱스 내용 조회 및 로그 저장 스크립트

이 스크립트는 Elasticsearch의 모든 인덱스를 조회하고,
각 인덱스의 메타데이터와 샘플 문서들을 로그 파일로 저장합니다.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, RequestError


class ElasticsearchIndexInspector:
    """Elasticsearch 인덱스 검사기"""
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        username: str = None,
        password: str = None,
        scheme: str = "http"
    ):
        """
        Elasticsearch 인덱스 검사기 초기화
        
        Args:
            host: ES 서버 호스트 (기본값: 환경변수 또는 3.35.110.161)
            port: ES 서버 포트 (기본값: 환경변수 또는 9200)
            username: 사용자명 (기본값: 환경변수 또는 elastic)
            password: 비밀번호 (기본값: 환경변수 또는 snomed)
            scheme: 연결 스키마 (기본값: http)
        """
        # 환경변수 또는 기본값 설정
        self.host = host or os.getenv("ES_SERVER_HOST", "3.35.110.161")
        self.port = port or int(os.getenv("ES_SERVER_PORT", "9200"))
        self.username = username or os.getenv("ES_SERVER_USERNAME", "elastic")
        self.password = password or os.getenv("ES_SERVER_PASSWORD", "snomed")
        self.scheme = scheme
        
        # 로그 설정
        self.setup_logging()
        
        # Elasticsearch 클라이언트 초기화
        self.es = self._create_elasticsearch_client()
        
        # 연결 테스트
        if not self._test_connection():
            raise ConnectionError("Elasticsearch 서버에 연결할 수 없습니다.")
            
        self.logger.info(f"Elasticsearch 연결 성공: {self.host}:{self.port}")
    
    def setup_logging(self):
        """로깅 설정"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"elasticsearch_index_inspection_{timestamp}.log"
        
        # 로거 설정
        self.logger = logging.getLogger('es_inspector')
        self.logger.setLevel(logging.INFO)
        
        # 파일 핸들러
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 핸들러 추가
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.log_filename = log_filename
        self.logger.info(f"로그 파일 생성: {log_filename}")
    
    def _create_elasticsearch_client(self) -> Elasticsearch:
        """Elasticsearch 클라이언트 생성"""
        try:
            # ES 8.x 방식으로 먼저 시도
            es = Elasticsearch(
                f"{self.scheme}://{self.host}:{self.port}",
                basic_auth=(self.username, self.password),
                request_timeout=60,
                retry_on_timeout=True,
                max_retries=3
            )
            return es
        except Exception as e:
            self.logger.warning(f"ES 8.x 방식 연결 실패, 대체 방식 시도: {e}")
            
            try:
                # ES 7.x 방식으로 대체 시도
                es = Elasticsearch(
                    hosts=[{"host": self.host, "port": self.port, "scheme": self.scheme}],
                    http_auth=(self.username, self.password),
                    request_timeout=60,
                    retry_on_timeout=True,
                    max_retries=3
                )
                return es
            except Exception as e2:
                self.logger.error(f"ES 7.x 방식도 실패: {e2}")
                raise ConnectionError(f"Elasticsearch 클라이언트 생성 실패: {e2}")
    
    def _test_connection(self) -> bool:
        """연결 테스트"""
        try:
            return self.es.ping()
        except Exception as e:
            self.logger.error(f"연결 테스트 실패: {e}")
            return False
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """클러스터 정보 조회"""
        try:
            info = self.es.info()
            self.logger.info("=== Elasticsearch 클러스터 정보 ===")
            self.logger.info(f"클러스터명: {info.get('cluster_name', 'Unknown')}")
            self.logger.info(f"버전: {info.get('version', {}).get('number', 'Unknown')}")
            self.logger.info(f"태그라인: {info.get('tagline', 'Unknown')}")
            return info
        except Exception as e:
            self.logger.error(f"클러스터 정보 조회 실패: {e}")
            return {}
    
    def get_all_indices(self) -> List[str]:
        """모든 인덱스 목록 조회"""
        try:
            indices = list(self.es.indices.get(index="*").keys())
            indices.sort()
            
            self.logger.info(f"=== 전체 인덱스 목록 (총 {len(indices)}개) ===")
            for i, index in enumerate(indices, 1):
                self.logger.info(f"{i:2d}. {index}")
            
            return indices
        except Exception as e:
            self.logger.error(f"인덱스 목록 조회 실패: {e}")
            return []
    
    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """인덱스 통계 정보 조회"""
        try:
            stats = self.es.indices.stats(index=index_name)
            index_stats = stats['indices'][index_name]
            
            # 주요 통계 정보 추출
            total_docs = index_stats['total']['docs']['count']
            total_size = index_stats['total']['store']['size_in_bytes']
            primary_size = index_stats['primaries']['store']['size_in_bytes']
            
            return {
                'total_documents': total_docs,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'primary_size_bytes': primary_size,
                'primary_size_mb': round(primary_size / (1024 * 1024), 2),
                'shards': index_stats['total']['docs']
            }
        except Exception as e:
            self.logger.warning(f"인덱스 {index_name} 통계 조회 실패: {e}")
            return {}
    
    def get_index_mapping(self, index_name: str) -> Dict[str, Any]:
        """인덱스 매핑 정보 조회"""
        try:
            mapping = self.es.indices.get_mapping(index=index_name)
            return mapping[index_name]['mappings']
        except Exception as e:
            self.logger.warning(f"인덱스 {index_name} 매핑 조회 실패: {e}")
            return {}
    
    def get_sample_documents(self, index_name: str, size: int = 5) -> List[Dict[str, Any]]:
        """인덱스의 샘플 문서들 조회"""
        try:
            response = self.es.search(
                index=index_name,
                body={
                    "query": {"match_all": {}},
                    "size": size
                }
            )
            
            documents = []
            for hit in response['hits']['hits']:
                documents.append({
                    'id': hit['_id'],
                    'source': hit['_source']
                })
            
            return documents
        except Exception as e:
            self.logger.warning(f"인덱스 {index_name} 샘플 문서 조회 실패: {e}")
            return []
    
    def inspect_index(self, index_name: str):
        """개별 인덱스 상세 조사"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"인덱스 상세 조사: {index_name}")
        self.logger.info(f"{'='*60}")
        
        # 1. 통계 정보
        stats = self.get_index_stats(index_name)
        if stats:
            self.logger.info(f"📊 통계 정보:")
            self.logger.info(f"  - 총 문서 수: {stats.get('total_documents', 'Unknown'):,}")
            self.logger.info(f"  - 총 크기: {stats.get('total_size_mb', 'Unknown')} MB")
            self.logger.info(f"  - 프라이머리 크기: {stats.get('primary_size_mb', 'Unknown')} MB")
        
        # 2. 매핑 정보
        mapping = self.get_index_mapping(index_name)
        if mapping:
            self.logger.info(f"🗺️  매핑 정보:")
            properties = mapping.get('properties', {})
            if properties:
                self.logger.info(f"  - 필드 수: {len(properties)}")
                self.logger.info(f"  - 필드 목록:")
                for field_name, field_info in properties.items():
                    field_type = field_info.get('type', 'unknown')
                    self.logger.info(f"    * {field_name}: {field_type}")
            else:
                self.logger.info("  - 매핑 정보 없음")
        
        # 3. 샘플 문서들
        sample_docs = self.get_sample_documents(index_name, size=3)
        if sample_docs:
            self.logger.info(f"📄 샘플 문서 ({len(sample_docs)}개):")
            for i, doc in enumerate(sample_docs, 1):
                self.logger.info(f"  --- 문서 {i} (ID: {doc['id']}) ---")
                # JSON을 예쁘게 출력
                doc_json = json.dumps(doc['source'], ensure_ascii=False, indent=4)
                for line in doc_json.split('\n'):
                    self.logger.info(f"  {line}")
        else:
            self.logger.info("📄 샘플 문서: 없음 또는 조회 실패")
        
        self.logger.info(f"{'='*60}\n")
    
    def inspect_all_indices(self):
        """모든 인덱스 조사"""
        self.logger.info("🔍 Elasticsearch 인덱스 전체 조사 시작")
        self.logger.info(f"조사 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 클러스터 정보
        cluster_info = self.get_cluster_info()
        
        # 인덱스 목록
        indices = self.get_all_indices()
        
        if not indices:
            self.logger.warning("조사할 인덱스가 없습니다.")
            return
        
        # 각 인덱스 상세 조사
        for i, index_name in enumerate(indices, 1):
            self.logger.info(f"\n진행 상황: {i}/{len(indices)} - {index_name}")
            try:
                self.inspect_index(index_name)
            except Exception as e:
                self.logger.error(f"인덱스 {index_name} 조사 중 오류 발생: {e}")
                continue
        
        self.logger.info("🎉 모든 인덱스 조사 완료!")
        self.logger.info(f"조사 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"로그 파일: {self.log_filename}")


def main():
    """메인 실행 함수"""
    try:
        # 인덱스 검사기 생성
        inspector = ElasticsearchIndexInspector()
        
        # 모든 인덱스 조사 실행
        inspector.inspect_all_indices()
        
        print(f"\n✅ 인덱스 조사가 완료되었습니다!")
        print(f"📄 로그 파일: {inspector.log_filename}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
