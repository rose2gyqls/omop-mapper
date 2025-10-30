"""
CONCEPT_RELATIONSHIP 데이터 인덱싱 스크립트

CONCEPT_RELATIONSHIP.csv 데이터를 읽어서 Elasticsearch에 인덱싱하는 스크립트입니다.
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm.auto import tqdm

# 로컬 모듈 임포트
from concept_relationship_data_processor import ConceptRelationshipDataProcessor
from elasticsearch_indexer import ConceptElasticsearchIndexer


class ConceptRelationshipIndexer:
    """CONCEPT_RELATIONSHIP 데이터 인덱서"""
    
    def __init__(
        self,
        csv_file_path: str,
        es_host: str = "3.35.110.161",
        es_port: int = 9200,
        es_username: str = "elastic",
        es_password: str = "snomed",
        index_name: str = "concept-relationship",
        chunk_size: int = 10000
    ):
        """
        인덱서 초기화
        
        Args:
            csv_file_path: CONCEPT_RELATIONSHIP.csv 파일 경로
            es_host: Elasticsearch 호스트
            es_port: Elasticsearch 포트
            es_username: Elasticsearch 사용자명
            es_password: Elasticsearch 비밀번호
            index_name: 인덱스명
            chunk_size: 데이터 처리 청크 크기
        """
        self.csv_file_path = csv_file_path
        self.index_name = index_name
        self.chunk_size = chunk_size
        
        # 컴포넌트 초기화
        logging.info("=== CONCEPT_RELATIONSHIP 인덱서 초기화 ===")
        
        # 1. 데이터 처리기 초기화
        logging.info("1. 데이터 처리기 초기화...")
        self.data_processor = ConceptRelationshipDataProcessor(csv_file_path)
        
        # 2. Elasticsearch 인덱서 초기화
        logging.info("2. Elasticsearch 인덱서 초기화...")
        self.es_indexer = ConceptElasticsearchIndexer(
            es_host=es_host,
            es_port=es_port,
            username=es_username,
            password=es_password,
            index_name=index_name,
            include_embeddings=False  # relationship 데이터는 임베딩 불필요
        )
        
        logging.info("=== 초기화 완료 ===")
    
    def run_full_indexing(
        self,
        delete_existing_index: bool = True,
        max_rows: int = None,
        skip_rows: int = 0
    ) -> bool:
        """
        전체 인덱싱 프로세스 실행
        
        Args:
            delete_existing_index: 기존 인덱스 삭제 여부
            max_rows: 최대 처리할 행 수 (None이면 전체)
            skip_rows: 건너뛸 행 수
            
        Returns:
            인덱싱 성공 여부
        """
        start_time = time.time()
        
        try:
            # 1. 인덱스 생성
            logging.info("=== 1단계: Elasticsearch 인덱스 생성 ===")
            if not self.es_indexer.create_index(delete_if_exists=delete_existing_index):
                logging.error("인덱스 생성 실패")
                return False
            
            # 2. 총 데이터 행 수 확인
            logging.info("=== 2단계: 데이터 크기 확인 ===")
            total_rows = self.data_processor.get_total_rows()
            logging.info(f"총 CONCEPT_RELATIONSHIP 데이터 행 수: {total_rows:,}")
            
            # 처리할 행 수 계산
            actual_skip = skip_rows
            actual_max = min(max_rows, total_rows - actual_skip) if max_rows else (total_rows - actual_skip)
            
            logging.info(f"건너뛸 행 수: {actual_skip:,}")
            logging.info(f"처리할 행 수: {actual_max:,}")
            
            # 3. 데이터 처리 및 인덱싱
            logging.info("=== 3단계: 데이터 처리 및 인덱싱 ===")
            
            total_processed = 0
            total_indexed = 0
            
            # 전체 진행률 표시를 위한 tqdm
            with tqdm(total=actual_max, desc="전체 진행률", unit="개") as pbar:
                
                # 청크 단위로 데이터 처리
                for chunk_df in self.data_processor.read_relationships_in_chunks(
                    chunk_size=self.chunk_size,
                    skip_rows=actual_skip,
                    max_rows=actual_max
                ):
                    
                    if len(chunk_df) == 0:
                        continue
                    
                    # Elasticsearch 문서 형식으로 변환
                    documents = self.data_processor.convert_to_elasticsearch_format(chunk_df)
                    
                    # Elasticsearch에 인덱싱
                    logging.info(f"청크 {len(documents)}개 문서 인덱싱 중...")
                    if self.es_indexer.index_concepts(
                        documents, 
                        show_progress=False
                    ):
                        total_indexed += len(documents)
                    
                    total_processed += len(chunk_df)
                    pbar.update(len(chunk_df))
                    
                    # 간단한 진행 상황 로깅
                    if total_processed % (self.chunk_size * 10) == 0:  # 10청크마다 로깅
                        elapsed_time = time.time() - start_time
                        rate = total_processed / elapsed_time if elapsed_time > 0 else 0
                        logging.info(
                            f"진행: {total_processed:,}/{actual_max:,} "
                            f"({total_processed/actual_max*100:.1f}%) | "
                            f"처리속도: {rate:.1f} rows/sec"
                        )
            
            # 4. 결과 확인
            logging.info("=== 4단계: 인덱싱 결과 확인 ===")
            
            # 인덱스 통계 확인
            stats = self.es_indexer.get_index_stats()
            logging.info(f"인덱스 통계: {stats}")
            
            # 총 소요 시간
            total_time = time.time() - start_time
            logging.info(f"총 소요 시간: {total_time/60:.1f}분")
            logging.info(f"평균 처리 속도: {total_processed/total_time:.1f} rows/sec")
            
            logging.info("=== 인덱싱 완료 ===")
            return True
            
        except Exception as e:
            logging.error(f"인덱싱 중 오류 발생: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False


def main(resume: bool = False):
    """메인 실행 함수"""
    
    index_name = "concept-relationship"
    print(f"=== {index_name} 인덱스 생성 ===")
    
    # 로깅 설정
    log_filename = f'concept_relationship_indexing_{time.strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    # 설정
    CSV_FILE_PATH = "/home/work/skku/hyo/omop-mapper/data/CONCEPT_RELATIONSHIP.csv"
    ES_HOST = "3.35.110.161"
    ES_PORT = 9200
    ES_USERNAME = "elastic"
    ES_PASSWORD = "snomed"
    CHUNK_SIZE = 10000  # relationship 데이터는 임베딩이 없으므로 큰 청크 크기 사용
    
    # 테스트용 설정 (실제 운영시에는 None으로 설정)
    MAX_ROWS = None  # None이면 전체 데이터 처리
    SKIP_ROWS = 0    # 건너뛸 행 수
    
    # Resume 기능: 기존 인덱스에서 현재 처리된 문서 수 확인
    if resume:
        try:
            # Elasticsearch 인덱서 임시 생성하여 현재 문서 수 확인
            from elasticsearch_indexer import ConceptElasticsearchIndexer
            temp_indexer = ConceptElasticsearchIndexer(
                es_host=ES_HOST,
                es_port=ES_PORT,
                username=ES_USERNAME,
                password=ES_PASSWORD,
                index_name=index_name,
                include_embeddings=False
            )
            
            # 현재 인덱스의 문서 수 확인
            stats = temp_indexer.get_index_stats()
            if stats and 'document_count' in stats:
                current_count = stats['document_count']
                SKIP_ROWS = current_count
                logging.info(f"🔄 Resume 모드: 현재 {current_count:,}개 문서가 인덱싱되어 있습니다.")
                logging.info(f"🔄 {current_count:,}개 문서를 건너뛰고 이후부터 처리를 시작합니다.")
            else:
                logging.warning("⚠️ Resume 모드이지만 기존 인덱스 정보를 가져올 수 없습니다. 처음부터 시작합니다.")
                logging.warning(f"⚠️ 받은 통계 정보: {stats}")
                SKIP_ROWS = 0
                
        except Exception as e:
            logging.error(f"❌ Resume 모드 설정 중 오류 발생: {e}")
            logging.warning("⚠️ 처음부터 시작합니다.")
            SKIP_ROWS = 0
    
    try:
        # 인덱서 초기화
        indexer = ConceptRelationshipIndexer(
            csv_file_path=CSV_FILE_PATH,
            es_host=ES_HOST,
            es_port=ES_PORT,
            es_username=ES_USERNAME,
            es_password=ES_PASSWORD,
            index_name=index_name,
            chunk_size=CHUNK_SIZE
        )
        
        # 전체 인덱싱 실행
        success = indexer.run_full_indexing(
            delete_existing_index=not resume,  # resume 모드일 때는 기존 인덱스 삭제하지 않음
            max_rows=MAX_ROWS,
            skip_rows=SKIP_ROWS
        )
        
        if success:
            logging.info(f"{index_name} 인덱싱이 성공적으로 완료되었습니다!")
        else:
            logging.error(f"{index_name} 인덱싱에 실패했습니다.")
            
    except Exception as e:
        logging.error(f"실행 중 오류 발생: {e}")
        import traceback
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    import sys
    import argparse
    
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description='CONCEPT_RELATIONSHIP 데이터 인덱싱')
    parser.add_argument('--resume', action='store_true', help='중단된 지점부터 인덱싱 재시작')
    
    args = parser.parse_args()
    
    # 메인 함수 실행
    main(resume=args.resume)

