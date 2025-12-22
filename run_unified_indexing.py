#!/usr/bin/env python3
"""
OMOP CDM Unified Indexing Script

Indexes OMOP CDM data into Elasticsearch from multiple sources.

Data Sources:
    - local_csv: Local CSV files (OMOP CDM vocabulary files)
    - postgres: PostgreSQL database (internal network CDM)
    - athena_api: OHDSI Athena API

Index Names:
    - concept
    - concept-relationship
    - concept-synonym

Usage:
    # Local CSV indexing
    python run_unified_indexing.py local_csv --data-folder /path/to/omop-cdm
    python run_unified_indexing.py local_csv --tables concept synonym

    # PostgreSQL indexing (default connection)
    python run_unified_indexing.py postgres
    python run_unified_indexing.py postgres --tables concept relationship

    # Athena API indexing
    python run_unified_indexing.py athena_api --vocabularies SNOMED ICD10CM

Options:
    --tables        Tables to index (concept, relationship, synonym)
    --gpu           GPU device number (default: 0)
    --no-embeddings Disable SapBERT embeddings
    --no-lowercase  Disable lowercase conversion
    --max-rows      Max rows to process (for testing)
    --resume        Keep existing index (don't delete)
"""

import sys
import argparse
import logging
import time
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "indexing"))


def setup_logging(source_type: str, gpu: int) -> str:
    """Configure logging with file and console output."""
    log_file = f'indexing_{source_type}_gpu{gpu}_{time.strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='OMOP CDM Unified Indexing',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required: data source type
    parser.add_argument(
        'source_type',
        choices=['local_csv', 'postgres', 'athena_api'],
        help='Data source type'
    )
    
    # Common options
    parser.add_argument('--tables', nargs='+',
        choices=['concept', 'relationship', 'synonym'],
        default=['concept', 'relationship', 'synonym'],
        help='Tables to index (default: all)')
    parser.add_argument('--gpu', type=int, default=0,
        help='GPU device number (default: 0, -1 for CPU)')
    parser.add_argument('--no-embeddings', action='store_true',
        help='Disable SapBERT embeddings')
    parser.add_argument('--no-lowercase', action='store_true',
        help='Disable lowercase conversion')
    parser.add_argument('--max-rows', type=int, default=None,
        help='Max rows to process (for testing)')
    parser.add_argument('--resume', action='store_true',
        help='Keep existing index (append mode)')
    parser.add_argument('--chunk-size', type=int, default=1000,
        help='Processing chunk size (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=128,
        help='Embedding batch size (default: 128)')
    
    # Elasticsearch options
    parser.add_argument('--es-host', default='3.35.110.161',
        help='Elasticsearch host')
    parser.add_argument('--es-port', type=int, default=9200,
        help='Elasticsearch port')
    parser.add_argument('--es-user', default='elastic',
        help='Elasticsearch username')
    parser.add_argument('--es-password', default='snomed',
        help='Elasticsearch password')
    
    # Local CSV options
    parser.add_argument('--data-folder',
        help='[local_csv] Path to CSV folder')
    parser.add_argument('--delimiter', default='\t',
        help='[local_csv] CSV delimiter (default: tab)')
    
    # PostgreSQL options
    parser.add_argument('--pg-host', default='172.23.100.146',
        help='[postgres] Database host')
    parser.add_argument('--pg-port', default='1341',
        help='[postgres] Database port')
    parser.add_argument('--pg-dbname', default='cdm_public',
        help='[postgres] Database name')
    parser.add_argument('--pg-user', default='cdmreader',
        help='[postgres] Database user')
    parser.add_argument('--pg-password', default='scdm2025!@',
        help='[postgres] Database password')
    
    # Athena API options
    parser.add_argument('--vocabularies', nargs='+',
        help='[athena_api] Vocabularies to fetch')
    parser.add_argument('--domains', nargs='+',
        help='[athena_api] Domains to fetch')
    parser.add_argument('--all-concepts', action='store_true',
        help='[athena_api] Include non-standard concepts')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging(args.source_type, args.gpu)
    
    print("=" * 70)
    print("OMOP CDM Unified Indexing")
    print("=" * 70)
    print(f"Source: {args.source_type}")
    print(f"Tables: {args.tables}")
    print(f"GPU: {args.gpu}")
    print(f"Embeddings: {'disabled' if args.no_embeddings else 'enabled'}")
    print(f"Log file: {log_file}")
    print("=" * 70)
    
    try:
        from indexing.unified_indexer import UnifiedIndexer, create_data_source
        
        # Create data source
        if args.source_type == 'local_csv':
            if not args.data_folder:
                args.data_folder = str(Path(__file__).parent / "data" / "omop-cdm")
                logging.info(f"Using default data folder: {args.data_folder}")
            
            data_source = create_data_source(
                'local_csv',
                data_folder=args.data_folder,
                delimiter=args.delimiter
            )
            
        elif args.source_type == 'postgres':
            data_source = create_data_source(
                'postgres',
                host=args.pg_host,
                port=args.pg_port,
                dbname=args.pg_dbname,
                user=args.pg_user,
                password=args.pg_password
            )
            
        elif args.source_type == 'athena_api':
            data_source = create_data_source(
                'athena_api',
                vocabularies=args.vocabularies,
                domains=args.domains,
                standard_only=not args.all_concepts
            )
        
        # Create indexer
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
            lowercase=not args.no_lowercase
        )
        
        # Run indexing
        results = indexer.index_all(
            delete_existing=not args.resume,
            max_rows=args.max_rows,
            tables=args.tables
        )
        
        # Print results
        print("\n" + "=" * 70)
        print("Results:")
        for table, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            print(f"  {table}: {status}")
        print("=" * 70)
        
        # Cleanup
        indexer.cleanup()
        
        if all(results.values()):
            print("\nAll indexing completed successfully!")
            return 0
        else:
            print("\nSome indexing failed. Check logs for details.")
            return 1
            
    except ImportError as e:
        print(f"\nImport error: {e}")
        print("Make sure all required packages are installed.")
        return 1
        
    except FileNotFoundError as e:
        print(f"\nFile not found: {e}")
        return 1
        
    except ConnectionError as e:
        print(f"\nConnection error: {e}")
        return 1
        
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
