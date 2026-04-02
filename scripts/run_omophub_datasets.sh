#!/usr/bin/env bash
# OMOPHub 배치: snomed-mapping-data-1000.csv, snuh-baseline-mapping-data.csv
# 사전 준비: OMOPHUB_API_KEY (https://dashboard.omophub.com)
# 문서: https://docs.omophub.com/introduction
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

if [[ -z "${OMOPHUB_API_KEY:-}" ]]; then
  echo "오류: OMOPHUB_API_KEY 가 설정되어 있지 않습니다." >&2
  echo "  ${ROOT}/.env 에 OMOPHUB_API_KEY=... 를 추가하거나 환경 변수로 export 하세요." >&2
  exit 1
fi

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

MODE="${OMOPHUB_MODE:-semantic}"
TOP_K="${OMOPHUB_TOP_K:-5}"
SLEEP="${OMOPHUB_SLEEP:-0}"

run_one() {
  local csv_path="$1"
  local label="$2"
  shift 2
  echo "=== OMOPHub (${MODE}) : ${label} ==="
  python -m omophub.csv_batch \
    --csv "${csv_path}" \
    --mode "${MODE}" \
    --top-k "${TOP_K}" \
    --sleep "${SLEEP}" \
    "$@"
}

# 필요 시 스크립트 상단 환경 변수로 조정:
#   OMOPHUB_MODE=basic
#   OMOPHUB_VOCAB='SNOMED'
#   OMOPHUB_THRESHOLD=0.5
EXTRA=()
if [[ -n "${OMOPHUB_VOCAB:-}" ]]; then
  EXTRA+=(--vocabulary-ids "${OMOPHUB_VOCAB}")
fi
if [[ -n "${OMOPHUB_THRESHOLD:-}" ]]; then
  EXTRA+=(--threshold "${OMOPHUB_THRESHOLD}")
fi

# set -u 일 때 빈 배열의 "${EXTRA[@]}" 가 Bash 4.4+ 에서 unbound 로 실패함 → 아래 패턴 사용
if ((${#EXTRA[@]} > 0)); then
  run_one "${ROOT}/data/snomed-mapping-data-1000.csv" "SNOMED 1000" "${EXTRA[@]}"
  run_one "${ROOT}/data/snuh-baseline-mapping-data.csv" "SNUH baseline" "${EXTRA[@]}"
else
  run_one "${ROOT}/data/snomed-mapping-data-1000.csv" "SNOMED 1000"
  run_one "${ROOT}/data/snuh-baseline-mapping-data.csv" "SNUH baseline"
fi

echo "완료. 결과는 ${ROOT}/outputs/omophub/ 디렉터리를 확인하세요."
