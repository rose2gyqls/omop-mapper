#!/usr/bin/env python3
"""consensus_run20 는 전체 빌드(JSON+baseline+필터)로만 재현합니다 — P/Y/A 시트만으로는 합의 표를 만들 수 없습니다."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "consensus_run20 은 run별 best_concept_id 와 (entity, concept) 인간 합의 맵 조회로 채워집니다. "
            "eval/build_*_run_matrix.py 를 실행하세요."
        )
    )
    ap.parse_args()
    print(
        "이 스크립트는 더 이상 P/Y/A 평균·다수결로 consensus_run20 을 만들지 않습니다. "
        "measurement/drug/eval 런 매트릭스 빌드 시 함께 생성되는 시트를 사용하세요.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
