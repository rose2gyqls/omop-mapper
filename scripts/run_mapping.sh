#!/bin/bash
# OMOP 매핑 실행 스크립트
#
# 데이터 소스 선택 시 기본 CSV 경로 및 전처리가 자동 적용됩니다.
# 출력: test_logs/mapping_{snuh|snomed}_{timestamp}.{json,log,xlsx}
#
# ============================================================================
# 매핑 옵션 설명
# ============================================================================
#
# [데이터 소스] 필수
#   snuh   : 기본 data/mapping_test_snuh_top10k.csv
#            전처리: vocabulary IN (SNOMED, LOINC)
#
#   snomed : 기본 data/mapping_test_snomed_no_note.csv
#            전처리: domain_id IN (Condition, Measurement, Drug, Observation, Procedure)
#
# [샘플링]
#   -n, --sample-size N   : 사용할 최대 샘플 수 (-n 미지정 시 전체 데이터). sample-per-domain 미사용 시 적용
#   --sample-per-domain N : 도메인별 N개씩 샘플. 예: --sample-per-domain 5
#   --random              : 랜덤 샘플링
#   --seed N              : 랜덤 시드 (기본: 42)
#
# [Scoring 모드] (ablation study용)
#   --scoring MODE
#     llm            : LLM 평가, 점수 미포함 (기본)
#     llm_with_score : LLM 평가, SapBERT 의미유사도 포함
#     semantic       : 의미유사도만
#     hybrid         : LLM + 의미유사도 혼합
#
# [병렬 처리]
#   -w, --workers N  : 워커 프로세스 수 (기본: 1). 4~8 권장 (메모리 ~1GB/워커)
#
# [반복 매핑] (일관성 검증)
#   -r, --repeat N   : 동일 데이터로 N회 매핑 (기본: 1). 5 입력 시 현황+5개 상세 시트 생성.
#
# [Validation]
#   기본: stage 1~3 점수 기반 최고 점수 매핑만 사용 (validation 미포함)
#   --validation      : LLM validation 모듈 포함. 출력: mapping_{snuh|snomed}_withval_{timestamp}.*
#
# ============================================================================
# 사용 예시
# ============================================================================
#
# SNUH 기본 (전체 데이터):
#   ./scripts/run_mapping.sh snuh
#
# SNUH 도메인별 5개씩 랜덤:
#   ./scripts/run_mapping.sh snuh --sample-per-domain 5 --random
#
# SNOMED 기본 (전체 데이터):
#   ./scripts/run_mapping.sh snomed
#
# SNOMED 도메인별 10개씩, semantic 모드:
#   ./scripts/run_mapping.sh snomed --sample-per-domain 10 --scoring semantic
#
# 병렬 4 워커 (1000건 이상 시 권장):
#   ./scripts/run_mapping.sh snuh --workers 4
#
# 5회 반복 (일관성 검증, 현황+5개 상세 시트):
#   ./scripts/run_mapping.sh snuh --repeat 5
#
# Validation 포함 실행 (with/without 비교용):
#   ./scripts/run_mapping.sh snuh --validation
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# 실시간 로그 출력 (버퍼링 비활성화)
export PYTHONUNBUFFERED=1

echo "============================================"
echo "OMOP 매핑 실행"
echo "============================================"
echo "프로젝트: $PROJECT_ROOT"
echo "============================================"

python run_mapping.py "$@"

echo ""
echo "============================================"
echo "완료! test_logs/ 에서 .json, .log, .xlsx 확인"
echo "============================================"
