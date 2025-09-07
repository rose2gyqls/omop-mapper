#!/usr/bin/env python3
"""
OMOP CONCEPT 데이터 Elasticsearch 인덱싱 메인 스크립트 (Eland 버전)

이 스크립트는 CONCEPT.csv 파일을 읽어서 Eland를 통해 SapBERT 모델을 
Elasticsearch에 배포하고 Ingest Pipeline을 사용하여 인덱싱하는 전체 파이프라인을 실행합니다.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional
import sys
import os
from dotenv import load_dotenv

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eland_model_manager import ElandModelManager
from elasticsearch_indexer import ConceptElasticsearchIndexer
from concept_data_processor import ConceptDataProcessor


class ConceptIndexingPipelineEland:
    """CONCEPT 데이터 인덱싱 파이프라인 (Eland 버전)"""
    
    def __init__(
        self,
        csv_file_path: str,
        es_host: str = "localhost",
        es_port: int = 9200,
        es_username: Optional[str] = None,
        es_password: Optional[str] = None,
        index_name: str = "concepts",
        sapbert_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        es_model_id: str = "sapbert-from-pubmedbert",
        pipeline_name: str = "concept-embedding-pipeline",
        chunk_size: int = 1000,
        indexing_batch_size: int = 500
    ):
        """
        인덱싱 파이프라인 초기화 (Eland 버전)
        
        Args:
            csv_file_path: CONCEPT.csv 파일 경로
            es_host: Elasticsearch 호스트
            es_port: Elasticsearch 포트  
            es_username: Elasticsearch 사용자명
            es_password: Elasticsearch 비밀번호
            index_name: Elasticsearch 인덱스명
            sapbert_model: SapBERT 모델명
            es_model_id: Elasticsearch에서 사용할 모델 ID
            pipeline_name: Ingest Pipeline 이름
            chunk_size: CSV 읽기 청크 크기
            indexing_batch_size: 인덱싱 배치 크기
        """
        self.csv_file_path = csv_file_path
        self.chunk_size = chunk_size
        self.indexing_batch_size = indexing_batch_size
        self.sapbert_model = sapbert_model
        self.es_model_id = es_model_id
        self.pipeline_name = pipeline_name
        
        # 컴포넌트 초기화
        logging.info("컴포넌트 초기화 중...")
        
        # 데이터 처리기
        self.data_processor = ConceptDataProcessor(csv_file_path)
        
        # Eland 모델 관리자
        self.model_manager = ElandModelManager(
            es_host=es_host,
            es_port=es_port,
            username=es_username,
            password=es_password
        )
        
        # Elasticsearch 인덱서
        self.es_indexer = ConceptElasticsearchIndexer(
            es_host=es_host,
            es_port=es_port,
            username=es_username,
            password=es_password,
            index_name=index_name
        )
        
        logging.info("컴포넌트 초기화 완료")
    
    def setup_model_and_pipeline(
        self,
        deploy_model: bool = True,
        create_pipeline: bool = True
    ) -> bool:
        """
        모델 배포 및 파이프라인 설정
        
        Args:
            deploy_model: 모델 배포 여부
            create_pipeline: 파이프라인 생성 여부
            
        Returns:
            설정 성공 여부
        """
        logging.info("=== 모델 및 파이프라인 설정 ===")
        
        # 1. 모델 배포
        if deploy_model:
            logging.info("SapBERT 모델 배포 중...")
            
            # 기존 모델 확인
            if self.model_manager.check_model_deployment(self.es_model_id):
                logging.info(f"✅ 모델이 이미 배포되어 있습니다: {self.es_model_id}")
            else:
                model_id = self.model_manager.deploy_sapbert_model(
                    hub_model_id=self.sapbert_model,
                    es_model_id=self.es_model_id,
                    start_deployment=True,
                    clear_previous=True
                )
                
                if not model_id:
                    logging.error("❌ 모델 배포 실패")
                    return False
                
                logging.info(f"✅ 모델 배포 성공: {model_id}")
            
            # 모델 추론 테스트
            if not self.model_manager.test_model_inference(self.es_model_id):
                logging.error("❌ 모델 추론 테스트 실패")
                return False
        
        # 2. Ingest Pipeline 생성
        if create_pipeline:
            logging.info("Ingest Pipeline 생성 중...")
            
            if not self.model_manager.create_ingest_pipeline(
                pipeline_name=self.pipeline_name,
                model_id=self.es_model_id,
                source_field="concept_name",
                target_field="concept_embedding"
            ):
                logging.error("❌ Ingest Pipeline 생성 실패")
                return False
            
            logging.info(f"✅ Ingest Pipeline 생성 성공: {self.pipeline_name}")
        
        return True
    
    def run_indexing(
        self,
        recreate_index: bool = True,
        max_rows: Optional[int] = None,
        skip_rows: int = 0,
        setup_ml: bool = True
    ) -> bool:
        """
        전체 인덱싱 파이프라인 실행
        
        Args:
            recreate_index: 인덱스 재생성 여부
            max_rows: 최대 처리할 행 수 (None이면 전체)
            skip_rows: 건너뛸 행 수
            setup_ml: ML 모델 및 파이프라인 설정 여부
            
        Returns:
            인덱싱 성공 여부
        """
        start_time = time.time()
        
        try:
            # 1. ML 모델 및 파이프라인 설정
            if setup_ml:
                if not self.setup_model_and_pipeline():
                    logging.error("모델 및 파이프라인 설정 실패")
                    return False
            
            # 2. 인덱스 생성
            logging.info("=== Elasticsearch 인덱스 생성 ===")
            if not self.es_indexer.create_index(delete_if_exists=recreate_index):
                logging.error("인덱스 생성 실패")
                return False
            
            # 3. 총 데이터 행 수 확인
            logging.info("=== 데이터 크기 확인 ===")
            total_rows = self.data_processor.get_total_rows()
            
            if max_rows:
                process_rows = min(total_rows - skip_rows, max_rows)
            else:
                process_rows = total_rows - skip_rows
                
            logging.info(f"총 데이터 행 수: {total_rows:,}")
            logging.info(f"처리할 행 수: {process_rows:,}")
            logging.info(f"건너뛸 행 수: {skip_rows:,}")
            
            # 4. 데이터 처리 및 인덱싱 (Ingest Pipeline 사용)
            logging.info("=== 데이터 처리 및 인덱싱 (Ingest Pipeline 사용) ===")
            
            processed_count = 0
            indexed_count = 0
            error_count = 0
            
            for chunk_idx, chunk_df in enumerate(
                self.data_processor.read_concepts_in_chunks(
                    chunk_size=self.chunk_size,
                    skip_rows=skip_rows,
                    max_rows=max_rows
                )
            ):
                chunk_start_time = time.time()
                
                try:
                    # 빈 청크 건너뛰기
                    if len(chunk_df) == 0:
                        continue
                    
                    logging.info(f"청크 {chunk_idx + 1} 처리 중: {len(chunk_df)}개 행")
                    
                    # Elasticsearch 문서 형식으로 변환 (임베딩 제외 - Pipeline에서 생성)
                    documents = self.data_processor.convert_to_elasticsearch_format(
                        chunk_df, 
                        embeddings=None,  # Ingest Pipeline에서 생성
                        include_embeddings=False  # Pipeline 사용
                    )
                    
                    # 배치 단위로 인덱싱 (Pipeline 적용)
                    for batch_start in range(0, len(documents), self.indexing_batch_size):
                        batch_docs = documents[batch_start:batch_start + self.indexing_batch_size]
                        
                        if self.es_indexer.index_concepts(
                            batch_docs, 
                            batch_size=len(batch_docs),
                            show_progress=False,
                            pipeline=self.pipeline_name  # Ingest Pipeline 적용
                        ):
                            indexed_count += len(batch_docs)
                        else:
                            error_count += len(batch_docs)
                            logging.error(f"배치 인덱싱 실패: {len(batch_docs)}개 문서")
                    
                    processed_count += len(chunk_df)
                    
                    # 진행률 및 성능 정보
                    chunk_time = time.time() - chunk_start_time
                    progress = (processed_count / process_rows) * 100
                    
                    logging.info(
                        f"청크 {chunk_idx + 1} 완료: "
                        f"{chunk_time:.1f}초, "
                        f"진행률: {progress:.1f}% "
                        f"({processed_count:,}/{process_rows:,})"
                    )
                    
                    # 메모리 정리
                    del documents, chunk_df
                    
                except Exception as e:
                    error_count += len(chunk_df)
                    logging.error(f"청크 {chunk_idx + 1} 처리 실패: {e}")
                    continue
            
            # 5. 결과 요약
            total_time = time.time() - start_time
            
            logging.info("=== 인덱싱 완료 ===")
            logging.info(f"총 처리 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
            logging.info(f"처리된 행 수: {processed_count:,}")
            logging.info(f"인덱싱된 문서 수: {indexed_count:,}")
            logging.info(f"오류 발생 문서 수: {error_count:,}")
            
            if processed_count > 0:
                logging.info(f"평균 처리 속도: {processed_count/total_time:.1f} 행/초")
            
            # 6. 인덱스 통계 확인
            stats = self.es_indexer.get_index_stats()
            if stats:
                logging.info(f"최종 인덱스 통계: {stats}")
            
            # 7. 임베딩 생성 확인
            self._verify_embeddings()
            
            return error_count == 0
            
        except Exception as e:
            logging.error(f"인덱싱 파이프라인 실행 실패: {e}")
            return False
    
    def _verify_embeddings(self):
        """임베딩 생성 확인"""
        try:
            # 몇 개 문서를 검색하여 임베딩이 생성되었는지 확인
            search_response = self.es_indexer.es.search(
                index=self.es_indexer.index_name,
                body={
                    "query": {"match_all": {}},
                    "size": 3,
                    "_source": ["concept_id", "concept_name", "concept_embedding"]
                }
            )
            
            embedding_count = 0
            for hit in search_response['hits']['hits']:
                doc = hit['_source']
                if 'concept_embedding' in doc and doc['concept_embedding']:
                    embedding_len = len(doc['concept_embedding'])
                    logging.info(f"✅ 임베딩 확인: {doc['concept_name']} -> 차원 {embedding_len}")
                    embedding_count += 1
                else:
                    logging.warning(f"⚠️ 임베딩 누락: {doc['concept_name']}")
            
            if embedding_count > 0:
                logging.info(f"✅ 임베딩 생성 성공 확인: {embedding_count}개 문서")
            else:
                logging.error("❌ 임베딩이 생성되지 않았습니다")
                
        except Exception as e:
            logging.error(f"임베딩 확인 실패: {e}")
    
    def cleanup_resources(self):
        """리소스 정리"""
        logging.info("=== 리소스 정리 ===")
        
        try:
            # Pipeline 삭제 (선택사항)
            # self.model_manager.delete_pipeline(self.pipeline_name)
            
            # 모델 삭제 (선택사항)
            # self.model_manager.delete_model(self.es_model_id)
            
            logging.info("✅ 리소스 정리 완료")
            
        except Exception as e:
            logging.warning(f"⚠️ 리소스 정리 중 오류: {e}")


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """로깅 설정"""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def main():
    """메인 함수"""
    # .env 파일 로드
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logging.info(f"✅ .env 파일 로드: {env_path}")
    else:
        logging.info("⚠️ .env 파일을 찾을 수 없습니다. 기본값 또는 명령행 인자를 사용합니다.")
    
    parser = argparse.ArgumentParser(
        description="OMOP CONCEPT 데이터를 Eland + SapBERT + Ingest Pipeline으로 Elasticsearch에 인덱싱"
    )
    
    # 필수 인자
    parser.add_argument(
        "--csv-file",
        required=True,
        help="CONCEPT.csv 파일 경로"
    )
    
    # Elasticsearch 설정 (.env 파일에서 기본값 읽기)
    parser.add_argument(
        "--es-host", 
        default=os.getenv("ES_SERVER_HOST", "localhost"), 
        help="Elasticsearch 호스트 (기본값: .env의 ES_SERVER_HOST 또는 localhost)"
    )
    parser.add_argument(
        "--es-port", 
        type=int, 
        default=int(os.getenv("ES_SERVER_PORT", "9200")), 
        help="Elasticsearch 포트 (기본값: .env의 ES_SERVER_PORT 또는 9200)"
    )
    parser.add_argument(
        "--es-username", 
        default=os.getenv("ES_SERVER_USERNAME"), 
        help="Elasticsearch 사용자명 (기본값: .env의 ES_SERVER_USERNAME)"
    )
    parser.add_argument(
        "--es-password", 
        default=os.getenv("ES_SERVER_PASSWORD"), 
        help="Elasticsearch 비밀번호 (기본값: .env의 ES_SERVER_PASSWORD)"
    )
    parser.add_argument("--index-name", default="concepts", help="Elasticsearch 인덱스명")
    
    # SapBERT 및 ML 설정
    parser.add_argument(
        "--sapbert-model",
        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        help="SapBERT 모델명"
    )
    parser.add_argument("--es-model-id", default="sapbert-from-pubmedbert", help="Elasticsearch 모델 ID")
    parser.add_argument("--pipeline-name", default="concept-embedding-pipeline", help="Ingest Pipeline 이름")
    
    # 처리 설정
    parser.add_argument("--chunk-size", type=int, default=1000, help="CSV 읽기 청크 크기")
    parser.add_argument("--indexing-batch-size", type=int, default=500, help="인덱싱 배치 크기")
    parser.add_argument("--max-rows", type=int, help="최대 처리할 행 수")
    parser.add_argument("--skip-rows", type=int, default=0, help="건너뛸 행 수")
    
    # 기타 설정
    parser.add_argument("--recreate-index", action="store_true", help="기존 인덱스 삭제 후 재생성")
    parser.add_argument("--skip-ml-setup", action="store_true", help="ML 모델 및 파이프라인 설정 건너뛰기")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", help="로그 파일 경로")
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(args.log_level, args.log_file)
    
    # .env 파일에서 로드된 설정 출력
    logging.info("=== 연결 설정 정보 ===")
    logging.info(f"Elasticsearch 호스트: {args.es_host}")
    logging.info(f"Elasticsearch 포트: {args.es_port}")
    logging.info(f"사용자명: {args.es_username or '없음'}")
    logging.info(f"비밀번호: {'설정됨' if args.es_password else '없음'}")
    logging.info(f"인덱스명: {args.index_name}")
    
    # CSV 파일 존재 확인
    if not Path(args.csv_file).exists():
        logging.error(f"CSV 파일을 찾을 수 없습니다: {args.csv_file}")
        return 1
    
    try:
        # 파이프라인 초기화
        pipeline = ConceptIndexingPipelineEland(
            csv_file_path=args.csv_file,
            es_host=args.es_host,
            es_port=args.es_port,
            es_username=args.es_username,
            es_password=args.es_password,
            index_name=args.index_name,
            sapbert_model=args.sapbert_model,
            es_model_id=args.es_model_id,
            pipeline_name=args.pipeline_name,
            chunk_size=args.chunk_size,
            indexing_batch_size=args.indexing_batch_size
        )
        
        # 인덱싱 실행
        success = pipeline.run_indexing(
            recreate_index=args.recreate_index,
            max_rows=args.max_rows,
            skip_rows=args.skip_rows,
            setup_ml=not args.skip_ml_setup
        )
        
        return 0 if success else 1
        
    except Exception as e:
        logging.error(f"프로그램 실행 실패: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
