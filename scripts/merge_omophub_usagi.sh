#!/usr/bin/env bash
# OMOPHub 요약 xlsx + USAGI CSV → comparison_SNOMED_omophub_usagi.xlsx, comparison_SNUH_omophub_usagi.xlsx
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

exec python -m omophub.merge_usagi_comparison "$@"
