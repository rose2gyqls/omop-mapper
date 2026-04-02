#!/usr/bin/env bash
# OMOPHub 출력 CSV에서 실패(429·rate limit 등) 행만 재요청해 병합 저장합니다.
# 사용: ./scripts/retry_omophub_output.sh outputs/omophub/xxx.csv
# 추가 인자는 python -m omophub.csv_batch 에 그대로 전달됩니다.
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
  exit 1
fi

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

FILE="${1:?사용법: $0 <outputs/omophub/...csv> [csv_batch 추가 옵션]}"
shift

exec python -m omophub.csv_batch --retry-from-output "${FILE}" "$@"
