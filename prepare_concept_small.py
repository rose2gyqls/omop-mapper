#!/usr/bin/env python3
"""
Concept-Small CSV 생성 스크립트

CONCEPT 테이블과 CONCEPT_SYNONYM 테이블(language_concept_id=4180186)을 합쳐서
concept-small.csv 파일을 생성합니다.

Usage:
    python prepare_concept_small.py --data-folder /path/to/omop-cdm
    python prepare_concept_small.py  # 기본 경로 사용
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


# 영어 동의어 language_concept_id
ENGLISH_LANGUAGE_CONCEPT_ID = 4180186


def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def create_concept_small(
    data_folder: str,
    output_file: str = "CONCEPT_SMALL.csv",
    delimiter: str = "\t"
) -> Path:
    """
    CONCEPT와 CONCEPT_SYNONYM을 합쳐서 CONCEPT_SMALL.csv 생성
    
    Args:
        data_folder: OMOP CDM 데이터 폴더 경로
        output_file: 출력 파일명
        delimiter: CSV 구분자
        
    Returns:
        생성된 파일 경로
    """
    logger = setup_logging()
    data_path = Path(data_folder)
    
    concept_path = data_path / "CONCEPT.csv"
    synonym_path = data_path / "CONCEPT_SYNONYM.csv"
    output_path = data_path / output_file
    
    logger.info("=" * 60)
    logger.info("Concept-Small CSV 생성 시작")
    logger.info("=" * 60)
    
    # 1. CONCEPT 테이블 로드
    logger.info(f"CONCEPT 테이블 로드: {concept_path}")
    if not concept_path.exists():
        raise FileNotFoundError(f"CONCEPT 파일을 찾을 수 없습니다: {concept_path}")
    
    concept = pd.read_csv(concept_path, sep=delimiter, low_memory=False, dtype=str)
    concept.columns = concept.columns.str.strip().str.lower()
    
    # concept_id를 정수로 통일 (문자열로 저장하되 비교를 위해)
    concept['concept_id'] = concept['concept_id'].astype(int)
    
    logger.info(f"  - 로드된 concept 수: {len(concept):,}")
    
    # 2. CONCEPT_SYNONYM 테이블 로드
    logger.info(f"CONCEPT_SYNONYM 테이블 로드: {synonym_path}")
    if not synonym_path.exists():
        raise FileNotFoundError(f"CONCEPT_SYNONYM 파일을 찾을 수 없습니다: {synonym_path}")
    
    syn = pd.read_csv(synonym_path, sep=delimiter, low_memory=False, dtype=str)
    syn.columns = syn.columns.str.strip().str.lower()
    syn['concept_id'] = syn['concept_id'].astype(int)
    syn['language_concept_id'] = syn['language_concept_id'].astype(int)
    
    logger.info(f"  - 로드된 전체 synonym 수: {len(syn):,}")
    
    # 3. 영어 동의어만 필터링 (language_concept_id = 4180186)
    syn = syn[syn['language_concept_id'] == ENGLISH_LANGUAGE_CONCEPT_ID]
    logger.info(f"  - 영어 synonym 수 (language_concept_id={ENGLISH_LANGUAGE_CONCEPT_ID}): {len(syn):,}")
    
    # 4. 원본 데이터에 name_type 추가
    concept['name_type'] = 'Original'
    
    # 5. 룩업 테이블 생성 (concept_id, concept_name, name_type 제외한 메타데이터)
    target_cols = [c for c in concept.columns if c not in ['concept_id', 'concept_name', 'name_type']]
    concept_lookup = concept.drop_duplicates('concept_id').set_index('concept_id')[target_cols]
    
    logger.info(f"  - 메타데이터 컬럼: {target_cols}")
    
    # 6. 동의어 행 생성
    syn_rows = syn[['concept_id', 'concept_synonym_name']].copy()
    syn_rows.rename(columns={'concept_synonym_name': 'concept_name'}, inplace=True)
    syn_rows['name_type'] = 'Synonym'
    
    # 7. 메타데이터 병합
    syn_rows = syn_rows.join(concept_lookup, on='concept_id')
    
    # 8. 최종 합치기
    # 컬럼 순서 맞추기
    final_columns = ['concept_id', 'concept_name', 'name_type'] + target_cols
    concept = concept[final_columns]
    syn_rows = syn_rows[final_columns]
    
    final_df = pd.concat([concept, syn_rows], ignore_index=True)
    
    logger.info(f"  - 최종 데이터 수: {len(final_df):,}")
    logger.info(f"    - Original: {len(final_df[final_df['name_type'] == 'Original']):,}")
    logger.info(f"    - Synonym: {len(final_df[final_df['name_type'] == 'Synonym']):,}")
    
    # 9. CSV로 저장
    logger.info(f"CSV 저장: {output_path}")
    final_df.to_csv(output_path, sep=delimiter, index=False)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"  - 파일 크기: {file_size_mb:.1f} MB")
    
    logger.info("=" * 60)
    logger.info("Concept-Small CSV 생성 완료")
    logger.info("=" * 60)
    
    return output_path


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='CONCEPT와 CONCEPT_SYNONYM을 합쳐서 CONCEPT_SMALL.csv 생성'
    )
    
    parser.add_argument(
        '--data-folder',
        default=str(Path(__file__).parent / "data" / "omop-cdm"),
        help='OMOP CDM 데이터 폴더 경로 (default: ./data/omop-cdm)'
    )
    parser.add_argument(
        '--output',
        default='CONCEPT_SMALL.csv',
        help='출력 파일명 (default: CONCEPT_SMALL.csv)'
    )
    parser.add_argument(
        '--delimiter',
        default='\t',
        help='CSV 구분자 (default: tab)'
    )
    
    args = parser.parse_args()
    
    try:
        output_path = create_concept_small(
            data_folder=args.data_folder,
            output_file=args.output,
            delimiter=args.delimiter
        )
        print(f"\n생성 완료: {output_path}")
        return 0
        
    except FileNotFoundError as e:
        print(f"\n오류: {e}")
        return 1
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
