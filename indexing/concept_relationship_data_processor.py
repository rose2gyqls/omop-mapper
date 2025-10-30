"""
CONCEPT_RELATIONSHIP CSV 데이터 처리기

CONCEPT_RELATIONSHIP.csv 파일을 읽고 Elasticsearch 인덱싱을 위해 데이터를 처리합니다.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional


class ConceptRelationshipDataProcessor:
    """CONCEPT_RELATIONSHIP CSV 데이터 처리기"""
    
    # CONCEPT_RELATIONSHIP CSV 컬럼
    RELATIONSHIP_COLUMNS = [
        "concept_id_1",
        "concept_id_2", 
        "relationship_id",
        "valid_start_date",
        "valid_end_date",
        "invalid_reason"
    ]
    
    def __init__(self, csv_file_path: str):
        """
        데이터 처리기 초기화
        
        Args:
            csv_file_path: CONCEPT_RELATIONSHIP.csv 파일 경로
        """
        self.csv_file_path = Path(csv_file_path)
        
        if not self.csv_file_path.exists():
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_file_path}")
        
        logging.info(f"CONCEPT_RELATIONSHIP 데이터 처리기 초기화: {csv_file_path}")
    
    def get_total_rows(self) -> int:
        """
        CSV 파일의 전체 행 수 반환 (헤더 제외)
        
        Returns:
            전체 데이터 행 수
        """
        try:
            # wc -l 명령을 사용하여 빠르게 행 수 계산
            import subprocess
            result = subprocess.run(
                ['wc', '-l', str(self.csv_file_path)],
                capture_output=True,
                text=True
            )
            total_lines = int(result.stdout.split()[0])
            # 헤더를 제외
            total_rows = total_lines - 1
            logging.info(f"총 데이터 행 수: {total_rows:,}")
            return total_rows
        except Exception as e:
            logging.warning(f"행 수 계산 중 오류 발생: {e}")
            # fallback: pandas로 계산 (느림)
            return sum(1 for _ in open(self.csv_file_path)) - 1
    
    def read_relationships_in_chunks(
        self,
        chunk_size: int = 10000,
        skip_rows: int = 0,
        max_rows: int = None
    ) -> Generator[pd.DataFrame, None, None]:
        """
        CONCEPT_RELATIONSHIP 데이터를 청크 단위로 읽기
        
        Args:
            chunk_size: 청크 크기
            skip_rows: 건너뛸 행 수
            max_rows: 최대 읽을 행 수 (None이면 전체)
            
        Yields:
            DataFrame 청크
        """
        try:
            # skiprows 설정
            skiprows = list(range(1, skip_rows + 1)) if skip_rows > 0 else None
            
            # 청크 단위로 읽기
            chunk_iterator = pd.read_csv(
                self.csv_file_path,
                sep='\t',  # 탭으로 구분
                chunksize=chunk_size,
                skiprows=skiprows,
                nrows=max_rows,
                dtype={
                    'concept_id_1': str,
                    'concept_id_2': str,
                    'relationship_id': str,
                    'valid_start_date': str,
                    'valid_end_date': str,
                    'invalid_reason': str
                },
                na_values=['', 'NA', 'NULL', 'null'],
                keep_default_na=True,
                low_memory=False
            )
            
            total_yielded = 0
            for chunk_df in chunk_iterator:
                # max_rows 제한 확인
                if max_rows and total_yielded >= max_rows:
                    break
                
                # 필요한 경우 청크 크기 조정
                if max_rows and total_yielded + len(chunk_df) > max_rows:
                    chunk_df = chunk_df.iloc[:max_rows - total_yielded]
                
                # 데이터 정제
                chunk_df = self._clean_data(chunk_df)
                
                total_yielded += len(chunk_df)
                yield chunk_df
                
        except Exception as e:
            logging.error(f"CSV 읽기 중 오류 발생: {e}")
            raise
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터 정제
        
        Args:
            df: 정제할 DataFrame
            
        Returns:
            정제된 DataFrame
        """
        # NaN 값을 None으로 변환
        df = df.replace({np.nan: None})
        
        # 날짜 형식 검증 및 변환
        for date_col in ['valid_start_date', 'valid_end_date']:
            if date_col in df.columns:
                df[date_col] = df[date_col].apply(self._validate_date_format)
        
        return df
    
    def _validate_date_format(self, date_str: str) -> Optional[str]:
        """
        날짜 형식 검증 및 변환 (YYYYMMDD)
        
        Args:
            date_str: 날짜 문자열
            
        Returns:
            유효한 날짜 문자열 또는 None
        """
        if pd.isna(date_str) or date_str is None:
            return None
        
        date_str = str(date_str).strip()
        
        # 이미 YYYYMMDD 형식인 경우
        if len(date_str) == 8 and date_str.isdigit():
            return date_str
        
        # 다른 형식 처리 (예: YYYY-MM-DD)
        try:
            from datetime import datetime
            # 다양한 형식 시도
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%Y%m%d']:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime('%Y%m%d')
                except ValueError:
                    continue
        except Exception:
            pass
        
        return None
    
    def convert_to_elasticsearch_format(
        self,
        df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        DataFrame을 Elasticsearch 인덱싱 형식으로 변환
        
        Args:
            df: 변환할 DataFrame
            
        Returns:
            Elasticsearch 문서 리스트
        """
        documents = []
        
        for idx, row in df.iterrows():
            doc = {}
            
            # 기본 필드 매핑
            for col in self.RELATIONSHIP_COLUMNS:
                if col in df.columns:
                    value = row[col]
                    # NaN을 None으로 변환
                    if pd.isna(value):
                        doc[col] = None
                    else:
                        doc[col] = str(value) if value is not None else None
            
            documents.append(doc)
        
        return documents
    
    def get_sample_data(self, n_samples: int = 100) -> pd.DataFrame:
        """
        샘플 데이터 반환
        
        Args:
            n_samples: 샘플 개수
            
        Returns:
            샘플 DataFrame
        """
        return pd.read_csv(
            self.csv_file_path,
            sep='\t',
            nrows=n_samples,
            dtype=str
        )


if __name__ == "__main__":
    # 간단한 테스트
    logging.basicConfig(level=logging.INFO)
    
    csv_file = "/home/work/skku/hyo/omop-mapper/data/CONCEPT_RELATIONSHIP.csv"
    processor = ConceptRelationshipDataProcessor(csv_file)
    
    # 샘플 데이터 확인
    sample_df = processor.get_sample_data(n_samples=10)
    print("\n샘플 데이터:")
    print(sample_df)
    
    # 총 행 수 확인
    total_rows = processor.get_total_rows()
    print(f"\n총 데이터 행 수: {total_rows:,}")

