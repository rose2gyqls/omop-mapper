#!/bin/bash
# OMOP CDM 인덱싱 스크립트
#
# ── 기본 사용법 ──
#   ./scripts/index.sh                                    # 기본 설정 (local_csv, concept-small/relationship/synonym)
#   ./scripts/index.sh local_csv                          # Local CSV 인덱싱
#   ./scripts/index.sh postgres                           # PostgreSQL 인덱싱
#   ./scripts/index.sh --prepare-only                     # CONCEPT_SMALL.csv 생성만
#
# ── 테이블 지정 (복수 가능) ──
#   ./scripts/index.sh local_csv --tables concept-small synonym
#   ./scripts/index.sh local_csv --tables concept-small   # concept-small만
#
# ── 끊긴 뒤 재시작 (Checkpoint 기반) ──
#   ./scripts/index.sh local_csv --resume                 # Checkpoint에서 마지막 성공 위치 읽어서 재개
#   ./scripts/index.sh local_csv --resume --tables synonym
#
# ── 429 완화 (bulk 요청 간 대기) ──
#   ./scripts/index.sh local_csv --resume --bulk-delay 1
#
# ── 테스트 (일부 행만) ──
#   ./scripts/index.sh local_csv --max-rows 10000
#
# ── 데이터 폴더 지정 ──
#   DATA_FOLDER=/path/to/csv ./scripts/index.sh local_csv
#   ./scripts/index.sh local_csv --data-folder /path/to/csv
#
# ── 안전성 보장 ──
#   - 멱등한 _id: 같은 데이터 재전송 시 덮어쓰기 (중복 없음)
#   - Checkpoint: 청크 단위로 진행 상황 기록, 실패 시 해당 청크부터 재시작
#   - 429 백오프: 5~300초 지수 백오프 재시도 (최대 7회)
#   - 개별 실패: bulk 응답 내 실패 문서 자동 재시도 (최대 3회)
#   - 검증: 완료 후 ES 문서 수 vs 원본 행 수 비교

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

# 인덱싱 실행 (DATA_FOLDER를 Python에 전달; 명령줄 --data-folder가 있으면 그쪽이 우선)
python run_indexing.py --data-folder "$DATA_FOLDER" "$@"

echo ""
echo "============================================"
echo "완료!"
echo "============================================"
