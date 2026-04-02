#!/usr/bin/env bash
# OMOPHub 출력 CSV → SNOMED / SNUH 요약 엑셀 (omophub_snomed_summary.xlsx, omophub_snuh_summary.xlsx)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

exec python -m omophub.export_results_excel --input-dir "${ROOT}/omophub/outputs" "$@"
