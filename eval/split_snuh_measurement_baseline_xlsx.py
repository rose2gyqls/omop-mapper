#!/usr/bin/env python3
"""
SNUH measurement comparison_baseline 통합 xlsx → 2개 산출 파일로 분리합니다.

1) entity + OMOPHub + USAGI + baseline (+ hub/usagi 일치 여부 is_same)
   — 시트 comparison_baseline, 인간 PHJ 평가점수 열 제외

2) 키 + 평가자 P/Y/A + 최종 + comment
   — 시트 평가점수_일치작업_평가자P만다름 전체

Usage:
  python eval/split_snuh_measurement_baseline_xlsx.py \\
    -i ~/Desktop/SNUH_measurement_평가자P_comparison_baseline.xlsx \\
    -o eval/out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SHEET_METHODS = "comparison_baseline"
SHEET_HUMAN = "평가점수_일치작업_평가자P만다름"


def load_methods_df(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=SHEET_METHODS, header=1)
    if "Unnamed: 15" in df.columns:
        df = df.rename(columns={"Unnamed: 15": "is_same"})
    drop_human = [c for c in df.columns if str(c).startswith("PHJ")]
    df = df.drop(columns=drop_human, errors="ignore")

    entity_cols = [
        "source_id",
        "domain_id",
        "source_value",
        "gt_concept_id",
        "gt_concept_name",
    ]
    hub_cols = [c for c in df.columns if c.startswith("omophub_")]
    usagi_cols = [c for c in df.columns if c.startswith("usagi_")]
    tail = [c for c in ("is_same", "baseline") if c in df.columns]

    ordered = [c for c in entity_cols + hub_cols + usagi_cols + tail if c in df.columns]
    missing = set(entity_cols + hub_cols + usagi_cols + tail) - set(df.columns)
    if missing:
        raise ValueError(f"{SHEET_METHODS}: 누락 컬럼 {sorted(missing)}")
    return df[ordered].copy()


def load_human_df(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=SHEET_HUMAN)
    expected = {"dataset", "entity_name", "domain_id", "concept", "P", "Y", "A", "최종", "comment"}
    if expected - set(df.columns):
        raise ValueError(f"{SHEET_HUMAN}: 예상 컬럼과 다름: {list(df.columns)}")
    return df


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    p = argparse.ArgumentParser(description="SNUH measurement baseline xlsx 2-way split")
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path.home() / "Desktop" / "SNUH_measurement_평가자P_comparison_baseline.xlsx",
    )
    p.add_argument("-o", "--output-dir", type=Path, default=root / "eval" / "out")
    args = p.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        print(f"입력 파일 없음: {inp}", file=sys.stderr)
        return 1

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out1 = out_dir / "SNUH_measurement_entity_baseline_hub_usagi.xlsx"
    out2 = out_dir / "SNUH_measurement_evaluators_P_Y_A_final.xlsx"

    mdf = load_methods_df(inp)
    hdf = load_human_df(inp)

    with pd.ExcelWriter(out1, engine="openpyxl") as w:
        mdf.to_excel(w, index=False, sheet_name="methods")
    with pd.ExcelWriter(out2, engine="openpyxl") as w:
        hdf.to_excel(w, index=False, sheet_name="human_scores")

    print(f"저장: {out1} ({len(mdf)}행)")
    print(f"저장: {out2} ({len(hdf)}행)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
