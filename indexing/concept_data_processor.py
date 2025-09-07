"""
CONCEPT CSV 데이터 처리기 모듈

OMOP CDM CONCEPT.csv 파일을 읽고 처리하는 기능을 제공합니다.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Iterator
import numpy as np
from pathlib import Path


class ConceptDataProcessor:
    """CONCEPT CSV 데이터 처리기"""
    
    # CONCEPT 테이블의 컬럼 정의
    CONCEPT_COLUMNS = [
        'concept_id',
        'concept_name', 
        'domain_id',
        'vocabulary_id',
        'concept_class_id',
        'standard_concept',
        'concept_code',
        'valid_start_date',
        'valid_end_date',
        'invalid_reason'
    ]
    
    def __init__(self, csv_file_path: str):
        """
        CONCEPT 데이터 처리기 초기화
        
        Args:
            csv_file_path: CONCEPT.csv 파일 경로
        """
        self.csv_file_path = Path(csv_file_path)
        
        if not self.csv_file_path.exists():
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_file_path}")
        
        logging.info(f"CONCEPT CSV 파일: {self.csv_file_path}")
        
        # 파일 크기 확인
        file_size_mb = self.csv_file_path.stat().st_size / (1024 * 1024)
        logging.info(f"파일 크기: {file_size_mb:.1f} MB")
    
    def get_total_rows(self) -> int:
        """
        CSV 파일의 총 행 수 계산 (헤더 제외)
        
        Returns:
            총 데이터 행 수
        """
        try:
            # 빠른 행 수 계산을 위해 wc 명령어 사용 (Unix 시스템)
            import subprocess
            result = subprocess.run(
                ["wc", "-l", str(self.csv_file_path)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                total_lines = int(result.stdout.split()[0])
                return max(0, total_lines - 1)  # 헤더 제외
        except:
            pass
        
        # fallback: pandas로 행 수 계산
        logging.info("pandas로 행 수 계산 중...")
        row_count = 0
        for chunk in pd.read_csv(
            self.csv_file_path,
            sep='\t',
            chunksize=10000,
            low_memory=False
        ):
            row_count += len(chunk)
        
        return row_count
    
    def read_concepts_in_chunks(
        self,
        chunk_size: int = 10000,
        skip_rows: int = 0,
        max_rows: Optional[int] = None
    ) -> Iterator[pd.DataFrame]:
        """
        CONCEPT 데이터를 청크 단위로 읽기
        
        Args:
            chunk_size: 청크 크기
            skip_rows: 건너뛸 행 수 (헤더 제외)
            max_rows: 최대 읽을 행 수
            
        Yields:
            pandas DataFrame 청크
        """
        try:
            # CSV 읽기 설정
            read_params = {
                'sep': '\t',
                'chunksize': chunk_size,
                'skiprows': skip_rows + 1 if skip_rows > 0 else None,  # 헤더는 항상 포함
                'nrows': max_rows,
                'low_memory': False,
                'dtype': {
                    'concept_id': str,
                    'concept_name': str,
                    'domain_id': str,
                    'vocabulary_id': str,
                    'concept_class_id': str,
                    'standard_concept': str,
                    'concept_code': str,
                    'valid_start_date': str,
                    'valid_end_date': str,
                    'invalid_reason': str
                },
                'na_values': ['', 'NULL', 'null', 'None'],
                'keep_default_na': True
            }
            
            # 첫 번째 청크에서 헤더 확인
            first_chunk = True
            
            for chunk_df in pd.read_csv(self.csv_file_path, **read_params):
                if first_chunk:
                    # 컬럼명 확인 및 정리
                    chunk_df.columns = chunk_df.columns.str.strip()
                    
                    # 필요한 컬럼이 있는지 확인
                    missing_cols = set(self.CONCEPT_COLUMNS) - set(chunk_df.columns)
                    if missing_cols:
                        logging.warning(f"누락된 컬럼: {missing_cols}")
                    
                    # 추가 컬럼 제거
                    available_cols = [col for col in self.CONCEPT_COLUMNS if col in chunk_df.columns]
                    chunk_df = chunk_df[available_cols]
                    
                    first_chunk = False
                    logging.info(f"컬럼 정보: {list(chunk_df.columns)}")
                
                # 데이터 정제
                chunk_df = self._clean_chunk_data(chunk_df)
                
                yield chunk_df
                
        except Exception as e:
            logging.error(f"CSV 파일 읽기 실패: {e}")
            raise
    
    def _clean_chunk_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        청크 데이터 정제
        
        Args:
            df: 정제할 DataFrame
            
        Returns:
            정제된 DataFrame
        """
        # 복사본 생성
        df = df.copy()
        
        # concept_id가 비어있는 행 제거
        df = df.dropna(subset=['concept_id'])
        
        # concept_id를 문자열로 변환
        df['concept_id'] = df['concept_id'].astype(str)
        
        # concept_name이 비어있는 행 제거
        df = df.dropna(subset=['concept_name'])
        
        # 텍스트 컬럼 정제
        text_columns = ['concept_name', 'domain_id', 'vocabulary_id', 'concept_class_id']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                # 빈 문자열을 None으로 변경
                df[col] = df[col].replace('', None)
        
        # 날짜 형식 검증 및 정제
        date_columns = ['valid_start_date', 'valid_end_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                # 8자리 날짜 형식이 아닌 경우 None으로 설정
                df[col] = df[col].apply(self._validate_date_format)
        
        # standard_concept 정제 (S, C, 또는 None만 허용)
        if 'standard_concept' in df.columns:
            df['standard_concept'] = df['standard_concept'].apply(
                lambda x: x if x in ['S', 'C'] else None
            )
        
        return df
    
    def _validate_date_format(self, date_str: str) -> Optional[str]:
        """
        날짜 형식 검증 (YYYYMMDD)
        
        Args:
            date_str: 날짜 문자열
            
        Returns:
            유효한 날짜 문자열 또는 None
        """
        if pd.isna(date_str) or date_str == 'None' or date_str == '':
            return None
        
        date_str = str(date_str).strip()
        
        # 8자리 숫자인지 확인
        if len(date_str) == 8 and date_str.isdigit():
            # 기본적인 날짜 유효성 검사
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            
            if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                return date_str
        
        return None
    
    def convert_to_elasticsearch_format(
        self,
        df: pd.DataFrame,
        embeddings: Optional[np.ndarray] = None,
        include_embeddings: bool = True
    ) -> List[Dict[str, Any]]:
        """
        DataFrame을 Elasticsearch 인덱싱 형식으로 변환
        
        Args:
            df: 변환할 DataFrame
            embeddings: 임베딩 배열 (선택사항)
            include_embeddings: 임베딩 포함 여부 (Ingest Pipeline 사용 시 False)
            
        Returns:
            Elasticsearch 문서 리스트
        """
        documents = []
        
        for idx, row in df.iterrows():
            doc = {}
            
            # 기본 필드 매핑
            for col in self.CONCEPT_COLUMNS:
                if col in df.columns:
                    value = row[col]
                    # NaN을 None으로 변환
                    if pd.isna(value):
                        doc[col] = None
                    else:
                        doc[col] = str(value) if value is not None else None
            
            # 임베딩 추가 (Ingest Pipeline을 사용하지 않는 경우에만)
            if include_embeddings and embeddings is not None and idx < len(embeddings):
                doc['concept_embedding'] = embeddings[idx].tolist()
            
            documents.append(doc)
        
        return documents
    
    def get_sample_data(self, n_samples: int = 100) -> pd.DataFrame:
        """
        샘플 데이터 추출
        
        Args:
            n_samples: 샘플 수
            
        Returns:
            샘플 DataFrame
        """
        try:
            sample_df = pd.read_csv(
                self.csv_file_path,
                sep='\t',
                nrows=n_samples,
                low_memory=False
            )
            
            # 컬럼명 정리
            sample_df.columns = sample_df.columns.str.strip()
            
            # 사용 가능한 컬럼만 선택
            available_cols = [col for col in self.CONCEPT_COLUMNS if col in sample_df.columns]
            sample_df = sample_df[available_cols]
            
            return self._clean_chunk_data(sample_df)
            
        except Exception as e:
            logging.error(f"샘플 데이터 추출 실패: {e}")
            raise


def test_concept_data_processor():
    """CONCEPT 데이터 처리기 테스트"""
    logging.basicConfig(level=logging.INFO)
    
    csv_path = "/Users/rose/Desktop/omop-mapper/data/CONCEPT.csv"
    
    try:
        # 데이터 처리기 초기화
        processor = ConceptDataProcessor(csv_path)
        
        # 총 행 수 확인
        total_rows = processor.get_total_rows()
        print(f"총 데이터 행 수: {total_rows:,}")
        
        # 샘플 데이터 확인
        sample_df = processor.get_sample_data(10)
        print(f"샘플 데이터 형태: {sample_df.shape}")
        print(f"컬럼: {list(sample_df.columns)}")
        print("\n샘플 데이터:")
        print(sample_df.head())
        
        # 청크 단위 읽기 테스트
        chunk_count = 0
        total_processed = 0
        
        for chunk_df in processor.read_concepts_in_chunks(chunk_size=1000, max_rows=5000):
            chunk_count += 1
            total_processed += len(chunk_df)
            
            if chunk_count <= 3:  # 처음 3개 청크만 출력
                print(f"\n청크 {chunk_count}: {len(chunk_df)}개 행")
                print(chunk_df.head(2))
        
        print(f"\n총 처리된 청크 수: {chunk_count}")
        print(f"총 처리된 행 수: {total_processed:,}")
        
    except Exception as e:
        print(f"테스트 실패: {e}")


if __name__ == "__main__":
    test_concept_data_processor()
