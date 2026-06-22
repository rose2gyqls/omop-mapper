#!/usr/bin/env python3
"""
OMOP CDM indexing script

Edit the configuration values and run, or override them with CLI options.

Usage:
    # Run with default settings (concept-small, synonym, relationship)
    python scripts/run_indexing.py
    
    # Run with CLI options
    python scripts/run_indexing.py local_csv --data-folder /path/to/data --tables concept-small synonym
    python scripts/run_indexing.py postgres --tables concept-small relationship synonym
    
    # Test (partial data only)
    python scripts/run_indexing.py local_csv --max-rows 10000
    
    # Resume from where it stopped (Checkpoint-based)
    python scripts/run_indexing.py local_csv --resume
    python scripts/run_indexing.py local_csv --resume --tables synonym
    
    # Add only 'Is a' relationships to an existing concept-relationship index (no deletion of existing data)
    python scripts/run_indexing.py local_csv --add-isa
    python scripts/run_indexing.py postgres --add-isa
    
    # Mitigate 429s (wait between bulk requests)
    python scripts/run_indexing.py local_csv --resume --bulk-delay 1
"""

import sys
import argparse
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)

# ============================================================================
# Default settings (used when no CLI option is given)
# ============================================================================

# Data source type: 'local_csv' or 'postgres'
DEFAULT_SOURCE = 'local_csv'

# List of tables to index
# Options: 'concept-small', 'synonym', 'relationship', 'concept'
DEFAULT_TABLES = ['concept-small', 'synonym', 'relationship']

# ----------------------------------------------------------------------------
# Local CSV settings
# ----------------------------------------------------------------------------
DEFAULT_DATA_FOLDER = str(Path(__file__).resolve().parent.parent / 'data' / 'omop-cdm')

# ----------------------------------------------------------------------------
# PostgreSQL settings
# ----------------------------------------------------------------------------
DEFAULT_PG_HOST = os.getenv('PG_HOST')
DEFAULT_PG_PORT = os.getenv('PG_PORT', '5432')
DEFAULT_PG_DBNAME = os.getenv('PG_DBNAME')
DEFAULT_PG_USER = os.getenv('PG_USER')
DEFAULT_PG_PASSWORD = os.getenv('PG_PASSWORD')

# ----------------------------------------------------------------------------
# Elasticsearch settings
# ----------------------------------------------------------------------------
DEFAULT_ES_HOST = os.getenv('ES_SERVER_HOST')
DEFAULT_ES_PORT = int(os.getenv('ES_SERVER_PORT', '9200'))
DEFAULT_ES_USER = os.getenv('ES_SERVER_USERNAME')
DEFAULT_ES_PASSWORD = os.getenv('ES_SERVER_PASSWORD')

# ----------------------------------------------------------------------------
# Indexing options (defaults tuned for large-scale / GPU)
# ----------------------------------------------------------------------------
DEFAULT_GPU = 0                 # GPU number (-1: use CPU)
DEFAULT_EMBEDDINGS = True       # Whether to include SapBERT embeddings
DEFAULT_LOWERCASE = True        # Whether to lowercase concept_name
DEFAULT_BATCH_SIZE = 512        # Embedding batch size (GPU: 512-1024 recommended)
DEFAULT_CHUNK_SIZE = 10000      # Data chunk size (larger = better GPU utilization, fewer ES round-trips)

# ============================================================================
# Main code
# ============================================================================

def setup_logging(source_type: str) -> str:
    """Configure logging"""
    log_file = f'indexing_{source_type}_{time.strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


def parse_args():
    """Parse CLI arguments"""
    parser = argparse.ArgumentParser(
        description='OMOP CDM Elasticsearch indexing',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data source (optional)
    parser.add_argument(
        'source_type', nargs='?',
        choices=['local_csv', 'postgres'],
        default=DEFAULT_SOURCE,
        help=f'Data source type (default: {DEFAULT_SOURCE})'
    )
    
    # Common options
    parser.add_argument('--tables', nargs='+',
        choices=['concept', 'concept-small', 'relationship', 'synonym'],
        default=DEFAULT_TABLES,
        help=f'Tables to index (default: {DEFAULT_TABLES})')
    parser.add_argument('--gpu', type=int, default=DEFAULT_GPU,
        help=f'GPU number, -1 for CPU (default: {DEFAULT_GPU})')
    parser.add_argument('--no-embeddings', action='store_true',
        help='Disable SapBERT embeddings')
    parser.add_argument('--no-lowercase', action='store_true',
        help='Disable lowercase conversion')
    parser.add_argument('--max-rows', type=int, default=None,
        help='Maximum number of rows to process (for testing)')
    parser.add_argument('--resume', action='store_true',
        help='Resume from where it stopped: read the last successful position '
             'from the checkpoint file (.indexing_checkpoint.json) and continue '
             'indexing from the next row. Keeps the existing index; idempotent _id avoids duplicates.')
    parser.add_argument('--add-isa', action='store_true',
        help="Add only 'Is a' relationships to an existing concept-relationship index. "
             "Never deletes existing data.")
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
        help=f'Embedding batch size (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
        help=f'Data chunk size (default: {DEFAULT_CHUNK_SIZE})')
    parser.add_argument('--bulk-delay', type=float, default=0.0,
        help='Wait time between bulk requests (seconds). Use when ES returns 429 (default: 0)')
    
    # Elasticsearch options
    parser.add_argument('--es-host', default=DEFAULT_ES_HOST,
        help='Elasticsearch host (default: ES_SERVER_HOST environment variable)')
    parser.add_argument('--es-port', type=int, default=DEFAULT_ES_PORT,
        help=f'Elasticsearch port (default: ES_SERVER_PORT environment variable or {DEFAULT_ES_PORT})')
    parser.add_argument('--es-user', default=DEFAULT_ES_USER,
        help='Elasticsearch user (default: ES_SERVER_USERNAME environment variable)')
    parser.add_argument('--es-password', default=DEFAULT_ES_PASSWORD,
        help='Elasticsearch password (default: ES_SERVER_PASSWORD environment variable)')
    
    # Local CSV options
    parser.add_argument('--data-folder', default=DEFAULT_DATA_FOLDER,
        help=f'CSV data folder (default: {DEFAULT_DATA_FOLDER})')
    
    # PostgreSQL options
    parser.add_argument('--pg-host', default=DEFAULT_PG_HOST,
        help='PostgreSQL host (default: PG_HOST environment variable)')
    parser.add_argument('--pg-port', default=DEFAULT_PG_PORT,
        help=f'PostgreSQL port (default: PG_PORT environment variable or {DEFAULT_PG_PORT})')
    parser.add_argument('--pg-dbname', default=DEFAULT_PG_DBNAME,
        help='PostgreSQL DB name (default: PG_DBNAME environment variable)')
    parser.add_argument('--pg-user', default=DEFAULT_PG_USER,
        help='PostgreSQL user (default: PG_USER environment variable)')
    parser.add_argument('--pg-password', default=DEFAULT_PG_PASSWORD,
        help='PostgreSQL password (default: PG_PASSWORD environment variable)')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Path setup
    _root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_root))
    sys.path.insert(0, str(_root / "indexing"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    
    # Logging setup
    log_file = setup_logging(args.source_type)
    logger = logging.getLogger(__name__)
    
    print("=" * 70)
    print("OMOP CDM Elasticsearch indexing")
    print("=" * 70)
    print(f"Data source: {args.source_type}")
    print(f"Tables: {args.tables}")
    print(f"Elasticsearch: {args.es_host}:{args.es_port}")
    print(f"GPU: {args.gpu}")
    print(f"Embeddings: {'disabled' if args.no_embeddings else 'enabled'}")
    if args.add_isa:
        print(f"Mode: ADD-ISA (add only 'Is a' relationships to existing concept-relationship, keep existing data)")
    elif args.resume:
        print(f"Mode: RESUME (Checkpoint-based resume)")
    else:
        print(f"Mode: FRESH (fresh indexing)")
    if args.bulk_delay > 0:
        print(f"Bulk delay: {args.bulk_delay}s (429 mitigation)")
    print(f"Log: {log_file}")
    print("=" * 70)
    
    try:
        from indexing.unified_indexer import UnifiedIndexer, create_data_source
        
        # 1. Create data source
        if args.source_type == 'local_csv':
            print(f"\nData folder: {args.data_folder}")
            
            # concept-small preprocessing (skipped in --add-isa mode)
            if 'concept-small' in args.tables and not args.add_isa:
                print("\n[1/2] Checking CONCEPT_SMALL.csv...")
                from prepare_concept_small import create_concept_small
                
                concept_small_path = Path(args.data_folder) / 'CONCEPT_SMALL.csv'
                if not concept_small_path.exists():
                    print("  -> Creating...")
                    create_concept_small(args.data_folder)
                else:
                    print("  -> Already exists (skip)")
            
            data_source = create_data_source(
                'local_csv',
                data_folder=args.data_folder
            )
            
        elif args.source_type == 'postgres':
            print(f"\nPostgreSQL: {args.pg_host}:{args.pg_port}/{args.pg_dbname}")
            print("\n[1/2] Connecting to PostgreSQL...")
            
            data_source = create_data_source(
                'postgres',
                host=args.pg_host,
                port=args.pg_port,
                dbname=args.pg_dbname,
                user=args.pg_user,
                password=args.pg_password
            )
            print("  -> Connected")
        
        # 2. Create and run the indexer
        print("\n[2/2] Elasticsearch indexing...")
        
        indexer = UnifiedIndexer(
            data_source=data_source,
            es_host=args.es_host,
            es_port=args.es_port,
            es_username=args.es_user,
            es_password=args.es_password,
            gpu_device=args.gpu,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            include_embeddings=not args.no_embeddings,
            lowercase=not args.no_lowercase,
            bulk_delay_sec=args.bulk_delay
        )
        
        if args.add_isa:
            # Add only 'Is a' relationships to existing concept-relationship (no deletion of existing data)
            success = indexer.index_relationships_isa_only(max_rows=args.max_rows)
            results = {'relationship(Is a added)': success}
        else:
            results = indexer.index_all(
                delete_existing=not args.resume,
                max_rows=args.max_rows,
                tables=args.tables
            )
        
        # 3. Print results
        print("\n" + "=" * 70)
        print("Results:")
        for table, success in results.items():
            status = "✓ Success" if success else "✗ Failed"
            print(f"  {table}: {status}")
        print("=" * 70)
        
        indexer.cleanup()
        
        if all(results.values()):
            print("\nDone! Indexing succeeded with no missing data.")
            return 0
        else:
            print("\nSome failures. Restart with --resume to continue from the failed part.")
            return 1
            
    except ImportError as e:
        print(f"\nError: {e}")
        print("Need to run pip install -r requirements.txt")
        return 1
    except FileNotFoundError as e:
        print(f"\nFile not found: {e}")
        return 1
    except ConnectionError as e:
        print(f"\nConnection failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\nError: {e}")
        print("Restart with --resume to continue from the failed part.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
