"""
SapBERT 임베딩을 포함한 CONCEPT_SYNONYM 데이터 인덱서

CONCEPT_SYNONYM.csv 파일을 읽어서 SapBERT 임베딩과 함께 Elasticsearch에 인덱싱합니다.
"""

import logging
import time
from datetime import datetime
from typing import Optional
import torch
from tqdm import tqdm

from concept_synonym_data_processor import ConceptSynonymDataProcessor
from elasticsearch_indexer import ConceptElasticsearchIndexer
from sapbert_embedder import SapBERTEmbedder


class ConceptSynonymIndexerWithSapBERT:
    """SapBERT 임베딩을 포함한 CONCEPT_SYNONYM 데이터 인덱서"""
    
    def __init__(
        self,
        csv_file_path: str,
        es_host: str = "3.35.110.161",
        es_port: int = 9200,
        es_username: str = "elastic",
        es_password: str = "snomed",
        index_name: str = "concept-synonym",
        model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        gpu_device: int = 0,
        batch_size: int = 128,
        chunk_size: int = 1000,
        lowercase_synonym_name: bool = False,
        include_embeddings: bool = True
    ):
        """
        CONCEPT_SYNONYM 인덱서 초기화
        
        Args:
            csv_file_path: CONCEPT_SYNONYM.csv 파일 경로
            es_host: Elasticsearch 호스트
            es_port: Elasticsearch 포트
            es_username: Elasticsearch 사용자명
            es_password: Elasticsearch 비밀번호
            index_name: 인덱스명
            model_name: SapBERT 모델명
            gpu_device: 사용할 GPU 번호
            batch_size: 임베딩 배치 크기
            chunk_size: 데이터 처리 청크 크기
            lowercase_synonym_name: synonym_name을 소문자로 변환할지 여부
            include_embeddings: 임베딩 포함 여부
        """
        self.csv_file_path = csv_file_path
        self.index_name = index_name
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.lowercase_synonym_name = lowercase_synonym_name
        self.include_embeddings = include_embeddings
        
        logging.info("=== CONCEPT_SYNONYM 인덱서 초기화 시작 ===")
        logging.info(f"CSV 파일: {csv_file_path}")
        logging.info(f"인덱스명: {index_name}")
        logging.info(f"GPU 디바이스: {gpu_device}")
        logging.info(f"배치 크기: {batch_size}")
        logging.info(f"청크 크기: {chunk_size}")
        logging.info(f"소문자 변환: {lowercase_synonym_name}")
        logging.info(f"임베딩 포함: {include_embeddings}")
        
        # 1. 데이터 처리기 초기화
        logging.info("=== 데이터 처리기 초기화 ===")
        self.data_processor = ConceptSynonymDataProcessor(csv_file_path)
        
        # 2. Elasticsearch 인덱서 초기화
        logging.info("=== Elasticsearch 인덱서 초기화 ===")
        self.es_indexer = ConceptElasticsearchIndexer(
            es_host=es_host,
            es_port=es_port,
            es_scheme="http",
            username=es_username,
            password=es_password,
            index_name=index_name,
            include_embeddings=include_embeddings
        )
        
        # 3. SapBERT 임베더 초기화 (임베딩이 필요한 경우에만)
        if include_embeddings:
            logging.info("=== SapBERT 임베더 초기화 ===")
            self.embedder = SapBERTEmbedder(
                model_name=model_name,
                device=f"cuda:{gpu_device}" if torch.cuda.is_available() else "cpu",
                batch_size=batch_size
            )
        else:
            self.embedder = None
            logging.info("임베딩 비활성화 - SapBERT 임베더 초기화 건너뜀")
        
        logging.info("=== 초기화 완료 ===")
    
    def run_full_indexing(
        self,
        delete_existing_index: bool = True,
        max_synonyms: int = None,
        skip_synonyms: int = 0
    ) -> bool:
        """
        전체 인덱싱 프로세스 실행
        
        Args:
            delete_existing_index: 기존 인덱스 삭제 여부
            max_synonyms: 최대 처리할 synonym 수 (None이면 전체)
            skip_synonyms: 건너뛸 synonym 수
            
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
            logging.info(f"총 CONCEPT_SYNONYM 데이터 행 수: {total_rows:,}")
            
            # 처리할 행 수 계산
            actual_skip = skip_synonyms
            actual_max = min(max_synonyms, total_rows - actual_skip) if max_synonyms else (total_rows - actual_skip)
            
            logging.info(f"건너뛸 행 수: {actual_skip:,}")
            logging.info(f"처리할 행 수: {actual_max:,}")
            
            # 3. 데이터 처리 및 인덱싱
            logging.info("=== 3단계: 데이터 처리 및 인덱싱 ===")
            
            total_processed = 0
            total_indexed = 0
            
            # 전체 진행률 표시를 위한 tqdm
            with tqdm(total=actual_max, desc="전체 진행률", unit="synonyms") as pbar:
                
                # 청크 단위로 데이터 처리
                for chunk_df in self.data_processor.read_synonyms_in_chunks(
                    chunk_size=self.chunk_size,
                    skip_rows=actual_skip,
                    max_rows=actual_max
                ):
                    if len(chunk_df) == 0:
                        continue
                    
                    chunk_start_time = time.time()
                    
                    # 임베딩 생성 (필요한 경우)
                    embeddings = None
                    if self.include_embeddings and self.embedder:
                        # concept_synonym_name 추출 (소문자 변환 옵션 적용)
                        synonym_names = chunk_df['concept_synonym_name'].tolist()
                        if self.lowercase_synonym_name:
                            synonym_names = [name.lower() if name else "" for name in synonym_names]
                        
                        # 임베딩 생성
                        embeddings = self.embedder.encode_batch(synonym_names)
                        logging.info(f"임베딩 생성 완료: {len(embeddings)} 개")
                    
                    # Elasticsearch 형식으로 변환
                    documents = self.data_processor.convert_to_elasticsearch_format(
                        chunk_df,
                        embeddings=embeddings,
                        include_embeddings=self.include_embeddings,
                        lowercase_synonym_name=self.lowercase_synonym_name
                    )
                    
                    # Elasticsearch에 인덱싱
                    success = self.es_indexer.index_concepts(
                        documents,
                        batch_size=min(500, len(documents)),  # synonym 데이터는 배치 크기를 작게
                        show_progress=False,
                        lowercase_concept_name=False  # 이미 처리됨
                    )
                    
                    if success:
                        total_indexed += len(documents)
                        logging.info(f"청크 인덱싱 완료: {len(documents)} 개")
                    else:
                        logging.error(f"청크 인덱싱 실패: {len(documents)} 개")
                    
                    total_processed += len(chunk_df)
                    pbar.update(len(chunk_df))
                    
                    chunk_time = time.time() - chunk_start_time
                    logging.info(f"청크 처리 시간: {chunk_time:.2f}초")
                    
                    # 메모리 정리
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # 4. 결과 요약
            total_time = time.time() - start_time
            logging.info("=== 인덱싱 완료 ===")
            logging.info(f"총 처리 시간: {total_time:.2f}초")
            logging.info(f"총 처리된 synonym 수: {total_processed:,}")
            logging.info(f"총 인덱싱된 synonym 수: {total_indexed:,}")
            logging.info(f"평균 처리 속도: {total_processed / total_time:.2f} synonyms/초")
            
            return total_indexed > 0
            
        except Exception as e:
            logging.error(f"인덱싱 중 오류 발생: {e}")
            return False
    
    def test_search(self, query: str = "blood pressure", top_k: int = 5):
        """
        인덱싱된 데이터로 검색 테스트
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
        """
        try:
            logging.info(f"=== 검색 테스트: '{query}' ===")
            
            # 기본 텍스트 검색
            text_results = self.es_indexer.search_concepts(
                query=query,
                size=top_k,
                search_type="text"
            )
            
            logging.info(f"텍스트 검색 결과 ({len(text_results)} 개):")
            for i, result in enumerate(text_results, 1):
                score = result.get('_score', 0)
                synonym_name = result['_source'].get('concept_synonym_name', 'N/A')
                concept_id = result['_source'].get('concept_id', 'N/A')
                logging.info(f"  {i}. [{score:.3f}] {synonym_name} (ID: {concept_id})")
            
            # 벡터 검색 (임베딩이 있는 경우)
            if self.include_embeddings and self.embedder:
                vector_results = self.es_indexer.search_concepts(
                    query=query,
                    size=top_k,
                    search_type="vector",
                    embedder=self.embedder
                )
                
                logging.info(f"벡터 검색 결과 ({len(vector_results)} 개):")
                for i, result in enumerate(vector_results, 1):
                    score = result.get('_score', 0)
                    synonym_name = result['_source'].get('concept_synonym_name', 'N/A')
                    concept_id = result['_source'].get('concept_id', 'N/A')
                    logging.info(f"  {i}. [{score:.3f}] {synonym_name} (ID: {concept_id})")
                    
        except Exception as e:
            logging.error(f"검색 테스트 실패: {e}")
    
    def cleanup(self):
        """리소스 정리"""
        logging.info("=== 리소스 정리 ===")
        
        if hasattr(self, 'embedder') and self.embedder:
            self.embedder.cleanup()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(
    create_small_index: bool = False, 
    gpu_device: int = 0, 
    resume: bool = False, 
    include_embeddings: bool = True
):
    """
    메인 실행 함수
    
    Args:
        create_small_index: 사용하지 않음 (synonym 인덱스는 항상 동일)
        gpu_device: 사용할 GPU 번호
        resume: 재시작 모드 (현재 미구현)
        include_embeddings: 임베딩 포함 여부
    """
    
    # 인덱스 설정
    index_name = "concept-synonym"
    lowercase_synonym_name = True  # synonym은 항상 소문자로 변환
    log_prefix = "concept_synonym"
    print(f"=== concept-synonym 인덱스 생성 (소문자 변환, GPU {gpu_device}) ===")
    
    # 로깅 설정
    log_filename = f'{log_prefix}_indexing_gpu{gpu_device}_{time.strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    # 설정
    CSV_FILE_PATH = "/home/work/skku/hyo/omop-mapper/data/CONCEPT_SYNONYM.csv"
    ES_HOST = "3.35.110.161"
    ES_PORT = 9200
    ES_USERNAME = "elastic"
    ES_PASSWORD = "snomed"
    MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    BATCH_SIZE = 128  # 임베딩 배치 크기
    CHUNK_SIZE = 1000   # 데이터 처리 청크 크기
    
    # 테스트용 설정 (실제 운영시에는 None으로 설정)
    MAX_SYNONYMS = None  # None이면 전체 데이터 처리
    SKIP_SYNONYMS = 0    # 건너뛸 synonym 수
    
    try:
        # 인덱서 초기화
        indexer = ConceptSynonymIndexerWithSapBERT(
            csv_file_path=CSV_FILE_PATH,
            es_host=ES_HOST,
            es_port=ES_PORT,
            es_username=ES_USERNAME,
            es_password=ES_PASSWORD,
            index_name=index_name,
            model_name=MODEL_NAME,
            gpu_device=gpu_device,
            batch_size=BATCH_SIZE,
            chunk_size=CHUNK_SIZE,
            lowercase_synonym_name=lowercase_synonym_name,
            include_embeddings=include_embeddings
        )
        
        # 전체 인덱싱 실행
        success = indexer.run_full_indexing(
            delete_existing_index=not resume,  # resume 모드일 때는 기존 인덱스 삭제하지 않음
            max_synonyms=MAX_SYNONYMS,
            skip_synonyms=SKIP_SYNONYMS
        )
        
        if success:
            logging.info(f"{index_name} 인덱싱이 성공적으로 완료되었습니다!")
            
            # 검색 테스트 실행
            indexer.test_search()
        else:
            logging.error(f"{index_name} 인덱싱에 실패했습니다.")
            
    except Exception as e:
        logging.error(f"실행 중 오류 발생: {e}")
    
    finally:
        # 리소스 정리
        if 'indexer' in locals():
            indexer.cleanup()


if __name__ == "__main__":
    import sys
    import argparse
    
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description='CONCEPT_SYNONYM 인덱스 생성')
    parser.add_argument('--gpu', type=int, default=0, help='사용할 GPU 번호 (기본값: 0)')
    parser.add_argument('--resume', action='store_true', help='중단된 지점부터 인덱싱 재시작')
    parser.add_argument('--no-embeddings', action='store_true', help='임베딩 없이 인덱싱')
    
    args = parser.parse_args()
    
    include_embeddings = not args.no_embeddings
    
    print("=" * 60)
    print(f"concept-synonym 인덱스 생성 시작 (GPU {args.gpu})")
    print("=" * 60)
    
    # 메인 함수 실행
    main(
        create_small_index=False,  # synonym 인덱스는 항상 동일
        gpu_device=args.gpu, 
        resume=args.resume,
        include_embeddings=include_embeddings
    )
    
    print("=" * 60)
    print(f"concept-synonym 인덱스 생성 완료")
    print("=" * 60)
