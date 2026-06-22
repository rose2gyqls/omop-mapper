#!/bin/bash
# OMOP mapping execution script
#
# Selecting a data source applies the default CSV path and preprocessing automatically.
# Output: test_logs/mapping_{snuh|snomed}_{timestamp}.{json,log,xlsx}
#
# ============================================================================
# Mapping option descriptions
# ============================================================================
#
# [Data source] required
#   snuh   : default data/mapping_test_snuh_top10k.csv
#            preprocessing: vocabulary IN (SNOMED, LOINC)
#
#   snomed : default data/mapping_test_snomed_no_note.csv
#            preprocessing: domain_id IN (Condition, Measurement, Drug, Observation, Procedure)
#
# [Sampling]
#   -n, --sample-size N   : Max number of samples to use (uses all data if -n not given). Applied when sample-per-domain is not used
#   --sample-per-domain N : Sample N per domain. Example: --sample-per-domain 5
#   --random              : Random sampling
#   --seed N              : Random seed (default: 42)
#
# [Scoring mode] (for ablation study)
#   --scoring MODE
#     llm            : LLM evaluation, score not included (default)
#     llm_with_score : LLM evaluation, SapBERT semantic similarity included
#     semantic       : semantic similarity only
#
# [LLM route selection]
#   --llm-provider   : openai | together
#   --llm-model      : Model name override (Together: gpt_oss_20b | mistral_small_24b | llama4_maverick aliases supported)
#   --llm-base-url   : OpenAI-compatible endpoint override
#   --llm-api-key-env: Name of the environment variable to read the API key from
#   --llm-temperature: temperature override
#   --llm-top-p      : top_p override
#   --llm-max-tokens : Max output tokens override
#
# [Parallel processing]
#   -w, --workers N  : Number of worker processes (default: 1). 4-8 recommended (~1GB memory per worker)
#
# [Repeat mapping] (consistency verification)
#   -r, --repeat N   : Map the same data N times (default: 1). Entering 5 generates a summary + 5 detail sheets.
#
# [Validation]
#   default: use only the highest-scoring mapping based on stage 1-3 scores (validation not included)
#   --validation      : Include the LLM validation module. Output: mapping_{snuh|snomed}_withval_{timestamp}.*
#
# ============================================================================
# Usage examples
# ============================================================================
#
# SNUH default (full data):
#   ./scripts/run_mapping.sh snuh
#
# SNUH 5 per domain, random:
#   ./scripts/run_mapping.sh snuh --sample-per-domain 5 --random
#
# SNOMED default (full data):
#   ./scripts/run_mapping.sh snomed
#
# SNOMED 10 per domain, semantic mode:
#   ./scripts/run_mapping.sh snomed --sample-per-domain 10 --scoring semantic
#
# Parallel 4 workers (recommended for 1000+ items):
#   ./scripts/run_mapping.sh snuh --workers 4
#
# Repeat 5 times (consistency verification, summary + 5 detail sheets):
#   ./scripts/run_mapping.sh snuh --repeat 5
#
# Run with validation included (for with/without comparison):
#   ./scripts/run_mapping.sh snuh --validation
#
# Run with Together GPT-OSS-20B:
#   ./scripts/run_mapping.sh snuh --llm-provider together --llm-model gpt_oss_20b --llm-api-key-env TOGETHER_API_KEY
#
# Run with Together Mistral Small 24B:
#   ./scripts/run_mapping.sh snuh --llm-provider together --llm-model mistral_small_24b --llm-api-key-env TOGETHER_API_KEY
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Real-time log output (disable buffering)
export PYTHONUNBUFFERED=1

echo "============================================"
echo "OMOP mapping execution"
echo "============================================"
echo "Project: $PROJECT_ROOT"
echo "============================================"

python scripts/run_mapping.py "$@"

echo ""
echo "============================================"
echo "Done! Check .json, .log, .xlsx in test_logs/"
echo "============================================"
