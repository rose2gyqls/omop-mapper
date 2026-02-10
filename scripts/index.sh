#!/bin/bash
# OMOP CDM 인덱싱 스크립트
#
# Usage:
#   ./scripts/index.sh                              # 기본 설정으로 실행
#   ./scripts/index.sh local_csv                    # Local CSV 인덱싱
#   ./scripts/index.sh postgres                     # PostgreSQL 인덱싱
#   ./scripts/index.sh local_csv --max-rows 10000   # 테스트
#   ./scripts/index.sh --prepare-only               # CONCEPT_SMALL.csv 생성만

set -e

# 스크립트 디렉토리 기준으로 프로젝트 루트 찾기
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# ============================================================================
# 설정 (필요시 수정)
# ============================================================================
DATA_FOLDER="${DATA_FOLDER:-/workspace/omop-mapper/data/omop-cdm}"

# ============================================================================
# 메인 로직
# ============================================================================

echo "============================================"
echo "OMOP CDM 인덱싱"
echo "============================================"
echo "프로젝트: $PROJECT_ROOT"
echo "데이터 폴더: $DATA_FOLDER"
echo "============================================"

# --prepare-only 옵션 체크
if [[ "$1" == "--prepare-only" ]]; then
    echo ""
    echo "[Step] CONCEPT_SMALL.csv 생성"
    echo "--------------------------------------------"
    python prepare_concept_small.py --data-folder "$DATA_FOLDER"
    echo ""
    echo "완료!"
    exit 0
fi

# 데이터 소스 확인 (첫 번째 인자 또는 기본값)
SOURCE_TYPE="${1:-local_csv}"

# local_csv인 경우 CONCEPT_SMALL.csv 생성
if [[ "$SOURCE_TYPE" == "local_csv" ]]; then
    CONCEPT_SMALL_PATH="$DATA_FOLDER/CONCEPT_SMALL.csv"
    
    echo ""
    echo "[Step 1/2] CONCEPT_SMALL.csv 확인"
    echo "--------------------------------------------"
    
    if [[ -f "$CONCEPT_SMALL_PATH" ]]; then
        echo "  -> 이미 존재: $CONCEPT_SMALL_PATH"
        echo "  -> 재생성하려면: ./scripts/index.sh --prepare-only"
    else
        echo "  -> 생성 중..."
        python prepare_concept_small.py --data-folder "$DATA_FOLDER"
        echo "  -> 완료"
    fi
    
    echo ""
    echo "[Step 2/2] Elasticsearch 인덱싱"
    echo "--------------------------------------------"
else
    echo ""
    echo "[Step] Elasticsearch 인덱싱 (PostgreSQL)"
    echo "--------------------------------------------"
fi

# 인덱싱 실행
python run_indexing.py "$@"

echo ""
echo "============================================"
echo "완료!"
echo "============================================"
