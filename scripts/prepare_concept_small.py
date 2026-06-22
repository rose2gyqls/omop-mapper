#!/usr/bin/env python3
"""
Concept-Small CSV generation script

Combines the CONCEPT table and the CONCEPT_SYNONYM table (language_concept_id=4180186)
to generate the concept-small.csv file.

Usage:
    python scripts/prepare_concept_small.py --data-folder /path/to/omop-cdm
    python scripts/prepare_concept_small.py  # use default path
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


# language_concept_id for English synonyms
ENGLISH_LANGUAGE_CONCEPT_ID = 4180186


def setup_logging():
    """Configure logging"""
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
    Combine CONCEPT and CONCEPT_SYNONYM to generate CONCEPT_SMALL.csv
    
    Args:
        data_folder: Path to the OMOP CDM data folder
        output_file: Output file name
        delimiter: CSV delimiter
        
    Returns:
        Path to the generated file
    """
    logger = setup_logging()
    data_path = Path(data_folder)
    
    concept_path = data_path / "CONCEPT.csv"
    synonym_path = data_path / "CONCEPT_SYNONYM.csv"
    output_path = data_path / output_file
    
    logger.info("=" * 60)
    logger.info("Starting Concept-Small CSV generation")
    logger.info("=" * 60)
    
    # 1. Load the CONCEPT table
    logger.info(f"Loading CONCEPT table: {concept_path}")
    if not concept_path.exists():
        raise FileNotFoundError(f"CONCEPT file not found: {concept_path}")
    
    concept = pd.read_csv(concept_path, sep=delimiter, low_memory=False, dtype=str)
    concept.columns = concept.columns.str.strip().str.lower()
    
    # Normalize concept_id to integer (stored as string, but for comparison)
    concept['concept_id'] = concept['concept_id'].astype(int)
    
    logger.info(f"  - Loaded concept count: {len(concept):,}")
    
    # 2. Load the CONCEPT_SYNONYM table
    logger.info(f"Loading CONCEPT_SYNONYM table: {synonym_path}")
    if not synonym_path.exists():
        raise FileNotFoundError(f"CONCEPT_SYNONYM file not found: {synonym_path}")
    
    syn = pd.read_csv(synonym_path, sep=delimiter, low_memory=False, dtype=str)
    syn.columns = syn.columns.str.strip().str.lower()
    syn['concept_id'] = syn['concept_id'].astype(int)
    syn['language_concept_id'] = syn['language_concept_id'].astype(int)
    
    logger.info(f"  - Total loaded synonym count: {len(syn):,}")
    
    # 3. Filter to English synonyms only (language_concept_id = 4180186)
    syn = syn[syn['language_concept_id'] == ENGLISH_LANGUAGE_CONCEPT_ID]
    logger.info(f"  - English synonym count (language_concept_id={ENGLISH_LANGUAGE_CONCEPT_ID}): {len(syn):,}")
    
    # 4. Add name_type to the original data
    concept['name_type'] = 'Original'
    
    # 5. Build a lookup table (metadata excluding concept_id, concept_name, name_type)
    target_cols = [c for c in concept.columns if c not in ['concept_id', 'concept_name', 'name_type']]
    concept_lookup = concept.drop_duplicates('concept_id').set_index('concept_id')[target_cols]
    
    logger.info(f"  - Metadata columns: {target_cols}")
    
    # 6. Build synonym rows
    syn_rows = syn[['concept_id', 'concept_synonym_name']].copy()
    syn_rows.rename(columns={'concept_synonym_name': 'concept_name'}, inplace=True)
    syn_rows['name_type'] = 'Synonym'
    
    # 7. Merge metadata
    syn_rows = syn_rows.join(concept_lookup, on='concept_id')
    
    # 8. Final concatenation
    # Align column order
    final_columns = ['concept_id', 'concept_name', 'name_type'] + target_cols
    concept = concept[final_columns]
    syn_rows = syn_rows[final_columns]
    
    final_df = pd.concat([concept, syn_rows], ignore_index=True)
    
    logger.info(f"  - Final row count: {len(final_df):,}")
    logger.info(f"    - Original: {len(final_df[final_df['name_type'] == 'Original']):,}")
    logger.info(f"    - Synonym: {len(final_df[final_df['name_type'] == 'Synonym']):,}")
    
    # 9. Save to CSV
    logger.info(f"Saving CSV: {output_path}")
    final_df.to_csv(output_path, sep=delimiter, index=False)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"  - File size: {file_size_mb:.1f} MB")
    
    logger.info("=" * 60)
    logger.info("Concept-Small CSV generation complete")
    logger.info("=" * 60)
    
    return output_path


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Combine CONCEPT and CONCEPT_SYNONYM to generate CONCEPT_SMALL.csv'
    )
    
    parser.add_argument(
        '--data-folder',
        default=str(Path(__file__).parent.parent / "data" / "omop-cdm"),
        help='Path to the OMOP CDM data folder (default: ./data/omop-cdm)'
    )
    parser.add_argument(
        '--output',
        default='CONCEPT_SMALL.csv',
        help='Output file name (default: CONCEPT_SMALL.csv)'
    )
    parser.add_argument(
        '--delimiter',
        default='\t',
        help='CSV delimiter (default: tab)'
    )
    
    args = parser.parse_args()
    
    try:
        output_path = create_concept_small(
            data_folder=args.data_folder,
            output_file=args.output,
            delimiter=args.delimiter
        )
        print(f"\nGeneration complete: {output_path}")
        return 0
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
