"""
CONCEPT 데이터 SapBERT 임베딩 인덱싱 메인 스크립트

CONCEPT.csv 데이터를 읽어서 SapBERT 모델로 임베딩을 생성하고
Elasticsearch에 인덱싱하는 통합 스크립트입니다.
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# 로컬 모듈 임포트
from concept_data_processor import ConceptDataProcessor
from sapbert_embedder import SapBERTEmbedder
from elasticsearch_indexer import ConceptElasticsearchIndexer


class ConceptIndexerWithSapBERT:
    """SapBERT 임베딩을 포함한 CONCEPT 데이터 인덱서"""
    
    def __init__(
        self,
        csv_file_path: str,
        es_host: str = "3.35.110.161",
        es_port: int = 9200,
        es_username: str = "elastic",
        es_password: str = "snomed",
        index_name: str = "concepts",
        model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        gpu_device: int = 0,
        batch_size: int = 128,
        chunk_size: int = 1000
    ):
        """
        인덱서 초기화
        
        Args:
            csv_file_path: CONCEPT.csv 파일 경로
            es_host: Elasticsearch 호스트
            es_port: Elasticsearch 포트
            es_username: Elasticsearch 사용자명
            es_password: Elasticsearch 비밀번호
            index_name: 인덱스명
            model_name: SapBERT 모델명
            gpu_device: 사용할 GPU 디바이스 번호
            batch_size: 임베딩 배치 크기
            chunk_size: 데이터 처리 청크 크기
        """
        self.csv_file_path = csv_file_path
        self.index_name = index_name
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        
        # GPU 디바이스 설정
        device = f"cuda:{gpu_device}" if gpu_device >= 0 else "cpu"
        
        # 컴포넌트 초기화
        logging.info("=== CONCEPT 인덱서 초기화 ===")
        
        # 1. 데이터 처리기 초기화
        logging.info("1. 데이터 처리기 초기화...")
        self.data_processor = ConceptDataProcessor(csv_file_path)
        
        # 2. SapBERT 임베딩 생성기 초기화
        logging.info("2. SapBERT 모델 로딩...")
        self.embedder = SapBERTEmbedder(
            model_name=model_name,
            device=device,
            batch_size=batch_size
        )
        
        # 3. Elasticsearch 인덱서 초기화
        logging.info("3. Elasticsearch 인덱서 초기화...")
        self.es_indexer = ConceptElasticsearchIndexer(
            es_host=es_host,
            es_port=es_port,
            username=es_username,
            password=es_password,
            index_name=index_name
        )
        
        logging.info("=== 초기화 완료 ===")
    
    def run_full_indexing(
        self,
        delete_existing_index: bool = True,
        max_concepts: int = None,
        skip_concepts: int = 0
    ) -> bool:
        """
        전체 인덱싱 프로세스 실행
        
        Args:
            delete_existing_index: 기존 인덱스 삭제 여부
            max_concepts: 최대 처리할 concept 수 (None이면 전체)
            skip_concepts: 건너뛸 concept 수
            
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
            logging.info(f"총 CONCEPT 데이터 행 수: {total_rows:,}")
            
            # 처리할 행 수 계산
            actual_skip = skip_concepts
            actual_max = min(max_concepts, total_rows - actual_skip) if max_concepts else (total_rows - actual_skip)
            
            logging.info(f"건너뛸 행 수: {actual_skip:,}")
            logging.info(f"처리할 행 수: {actual_max:,}")
            
            # 3. 데이터 처리 및 인덱싱
            logging.info("=== 3단계: 데이터 처리 및 인덱싱 ===")
            
            total_processed = 0
            total_indexed = 0
            
            # 전체 진행률 표시를 위한 tqdm
            with tqdm(total=actual_max, desc="전체 진행률", unit="개") as pbar:
                
                # 청크 단위로 데이터 처리
                for chunk_df in self.data_processor.read_concepts_in_chunks(
                    chunk_size=self.chunk_size,
                    skip_rows=actual_skip,
                    max_rows=actual_max
                ):
                    
                    if len(chunk_df) == 0:
                        continue
                    
                    # concept_name 추출 (임베딩용)
                    concept_names = chunk_df['concept_name'].fillna('').tolist()
                    
                    # SapBERT 임베딩 생성
                    logging.info(f"청크 {len(chunk_df)}개 concept 임베딩 생성 중...")
                    embeddings = self.embedder.encode_texts(concept_names, show_progress=False)
                    
                    # Elasticsearch 문서 형식으로 변환
                    documents = self.data_processor.convert_to_elasticsearch_format(
                        chunk_df, 
                        embeddings=embeddings,
                        include_embeddings=True
                    )
                    
                    # Elasticsearch에 인덱싱
                    logging.info(f"청크 {len(documents)}개 문서 인덱싱 중...")
                    if self.es_indexer.index_concepts(documents, show_progress=False):
                        total_indexed += len(documents)
                    
                    total_processed += len(chunk_df)
                    pbar.update(len(chunk_df))
                    
                    # 진행 상황 로깅
                    elapsed_time = time.time() - start_time
                    rate = total_processed / elapsed_time if elapsed_time > 0 else 0
                    
                    if total_processed > 0:
                        remaining_concepts = actual_max - total_processed
                        estimated_time_remaining = remaining_concepts / rate if rate > 0 else 0
                        
                        logging.info(
                            f"진행: {total_processed:,}/{actual_max:,} "
                            f"({total_processed/actual_max*100:.1f}%) | "
                            f"처리속도: {rate:.1f} concepts/sec | "
                            f"예상 남은 시간: {estimated_time_remaining/60:.1f}분"
                        )
            
            # 4. 결과 확인
            logging.info("=== 4단계: 인덱싱 결과 확인 ===")
            
            # 인덱스 통계 확인
            stats = self.es_indexer.get_index_stats()
            logging.info(f"인덱스 통계: {stats}")
            
            # 총 소요 시간
            total_time = time.time() - start_time
            logging.info(f"총 소요 시간: {total_time/60:.1f}분")
            logging.info(f"평균 처리 속도: {total_processed/total_time:.1f} concepts/sec")
            
            logging.info("=== 인덱싱 완료 ===")
            return True
            
        except Exception as e:
            logging.error(f"인덱싱 중 오류 발생: {e}")
            return False
    
    def test_search(self, test_queries: List[str] = None) -> None:
        """
        인덱싱된 데이터로 검색 테스트
        
        Args:
            test_queries: 테스트할 쿼리 리스트
        """
        if test_queries is None:
            test_queries = [
                "covid-19",
                "hypertension",
                "diabetes",
                "heart failure",
                "pneumonia"
            ]
        
        logging.info("=== 검색 테스트 ===")
        
        for query in test_queries:
            logging.info(f"\n검색 쿼리: '{query}'")
            
            # 쿼리 임베딩 생성
            query_embedding = self.embedder.encode_texts([query], show_progress=False)[0]
            
            # 유사도 검색
            results = self.es_indexer.search_by_embedding(
                query_embedding.tolist(),
                size=5,
                min_score=0.5
            )
            
            # 결과 출력
            for i, result in enumerate(results, 1):
                logging.info(
                    f"  {i}. {result['concept_name']} "
                    f"(ID: {result['concept_id']}, "
                    f"유사도: {result['similarity_score']:.3f})"
                )
    
    def cleanup(self):
        """리소스 정리"""
        if hasattr(self, 'embedder'):
            del self.embedder
        
        # GPU 메모리 정리
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """메인 실행 함수"""
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'concept_indexing_{time.strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    
    # 설정
    CSV_FILE_PATH = "/home/work/skku/hyo/omop-mapper/data/CONCEPT.csv"
    ES_HOST = "3.35.110.161"
    ES_PORT = 9200
    ES_USERNAME = "elastic"
    ES_PASSWORD = "snomed"
    INDEX_NAME = "concepts"
    MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    GPU_DEVICE = 0  # GPU 0번 사용
    BATCH_SIZE = 128  # 임베딩 배치 크기
    CHUNK_SIZE = 1000  # 데이터 처리 청크 크기
    
    # 테스트용 설정 (실제 운영시에는 None으로 설정)
    MAX_CONCEPTS = None  # None이면 전체 데이터 처리
    SKIP_CONCEPTS = 0    # 건너뛸 concept 수
    
    try:
        # 인덱서 초기화
        indexer = ConceptIndexerWithSapBERT(
            csv_file_path=CSV_FILE_PATH,
            es_host=ES_HOST,
            es_port=ES_PORT,
            es_username=ES_USERNAME,
            es_password=ES_PASSWORD,
            index_name=INDEX_NAME,
            model_name=MODEL_NAME,
            gpu_device=GPU_DEVICE,
            batch_size=BATCH_SIZE,
            chunk_size=CHUNK_SIZE
        )
        
        # 전체 인덱싱 실행
        success = indexer.run_full_indexing(
            delete_existing_index=True,
            max_concepts=MAX_CONCEPTS,
            skip_concepts=SKIP_CONCEPTS
        )
        
        if success:
            logging.info("인덱싱이 성공적으로 완료되었습니다!")
            
            # 검색 테스트 실행
            indexer.test_search()
        else:
            logging.error("인덱싱에 실패했습니다.")
            
    except Exception as e:
        logging.error(f"실행 중 오류 발생: {e}")
    
    finally:
        # 리소스 정리
        if 'indexer' in locals():
            indexer.cleanup()


if __name__ == "__main__":
    main()
