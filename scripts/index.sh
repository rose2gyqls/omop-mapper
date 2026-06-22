#!/bin/bash
# OMOP CDM indexing script
#
# ── Basic usage ──
#   ./scripts/index.sh                                    # Default settings (local_csv, concept-small/relationship/synonym)
#   ./scripts/index.sh local_csv                          # Local CSV indexing
#   ./scripts/index.sh postgres                           # PostgreSQL indexing
#   ./scripts/index.sh --prepare-only                     # Generate CONCEPT_SMALL.csv only
#
# ── Specifying tables (multiple allowed) ──
#   ./scripts/index.sh local_csv --tables concept-small synonym
#   ./scripts/index.sh local_csv --tables concept-small   # concept-small only
#
# ── Add only 'Is a' relationships to existing concept-relationship (no deletion of existing data) ──
#   ./scripts/index.sh local_csv --add-isa
#   ./scripts/index.sh postgres --add-isa
#
# ── Restart after an interruption (Checkpoint-based) ──
#   ./scripts/index.sh local_csv --resume                 # Read the last successful position from the checkpoint and resume
#   ./scripts/index.sh local_csv --resume --tables synonym
#
# ── Mitigate 429s (wait between bulk requests) ──
#   ./scripts/index.sh local_csv --resume --bulk-delay 1
#
# ── Test (partial rows only) ──
#   ./scripts/index.sh local_csv --max-rows 10000
#
# ── Specifying the data folder ──
#   DATA_FOLDER=/path/to/csv ./scripts/index.sh local_csv
#   ./scripts/index.sh local_csv --data-folder /path/to/csv
#
# ── Safety guarantees ──
#   - Idempotent _id: re-sending the same data overwrites (no duplicates)
#   - Checkpoint: record progress per chunk, restart from that chunk on failure
#   - 429 backoff: exponential backoff retry of 5-300s (up to 7 times)
#   - Individual failures: automatically retry failed documents within a bulk response (up to 3 times)
#   - Verification: after completion, compare ES document count vs source row count

set -e

# Find the project root relative to the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# ============================================================================
# Settings (edit if needed)
# ============================================================================
DATA_FOLDER="${DATA_FOLDER:-$PROJECT_ROOT/data/omop-cdm}"

# ============================================================================
# Main logic
# ============================================================================

echo "============================================"
echo "OMOP CDM indexing"
echo "============================================"
echo "Project: $PROJECT_ROOT"
echo "Data folder: $DATA_FOLDER"
echo "============================================"

# Check the --prepare-only option
if [[ "$1" == "--prepare-only" ]]; then
    echo ""
    echo "[Step] Generate CONCEPT_SMALL.csv"
    echo "--------------------------------------------"
    python scripts/prepare_concept_small.py --data-folder "$DATA_FOLDER"
    echo ""
    echo "Done!"
    exit 0
fi

# Determine the data source (first argument or default)
SOURCE_TYPE="${1:-local_csv}"

# --add-isa: CONCEPT_SMALL not needed (uses CONCEPT_RELATIONSHIP only)
ADD_ISA=false
for arg in "$@"; do [[ "$arg" == "--add-isa" ]] && ADD_ISA=true && break; done

# For local_csv, generate CONCEPT_SMALL.csv (skipped in --add-isa mode)
if [[ "$SOURCE_TYPE" == "local_csv" && "$ADD_ISA" != "true" ]]; then
    CONCEPT_SMALL_PATH="$DATA_FOLDER/CONCEPT_SMALL.csv"
    
    echo ""
    echo "[Step 1/2] Check CONCEPT_SMALL.csv"
    echo "--------------------------------------------"
    
    if [[ -f "$CONCEPT_SMALL_PATH" ]]; then
        echo "  -> Already exists: $CONCEPT_SMALL_PATH"
        echo "  -> To regenerate: ./scripts/index.sh --prepare-only"
    else
        echo "  -> Creating..."
        python scripts/prepare_concept_small.py --data-folder "$DATA_FOLDER"
        echo "  -> Done"
    fi
    
    echo ""
    echo "[Step 2/2] Elasticsearch indexing"
    echo "--------------------------------------------"
elif [[ "$SOURCE_TYPE" == "local_csv" && "$ADD_ISA" == "true" ]]; then
    echo ""
    echo "[Step] Add 'Is a' relationships to concept-relationship"
    echo "--------------------------------------------"
else
    echo ""
    echo "[Step] Elasticsearch indexing (PostgreSQL)"
    echo "--------------------------------------------"
fi

# Run indexing (pass DATA_FOLDER to Python; a command-line --data-folder takes precedence)
python scripts/run_indexing.py --data-folder "$DATA_FOLDER" "$@"

echo ""
echo "============================================"
echo "Done!"
echo "============================================"
