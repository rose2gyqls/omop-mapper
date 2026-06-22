#!/usr/bin/env python3
"""
SNUH Drug comparison_baseline 통합 xlsx → 2개 산출 파일.

1) entity + OMOPHub + USAGI + baseline + PHJ 평가점수(허브/USAGI 열 모두 유지)
   — 시트 comparison_baseline, header=1

2) 평가점수_일치작업_평가자P만다름 전체 (P, Y, A, 최종, 코멘트 등)

Usage:
  python eval/split_snuh_drug_baseline_xlsx.py \\
    -i eval/SNUH_drug_평가자P_comparison_baseline.xlsx \\
    -o eval/out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SHEET_METHODS = "comparison_baseline"
SHEET_HUMAN = "평가점수_일치작업_평가자P만다름"


def load_methods_df_simple(path: Path) -> pd.DataFrame:
    """PHJ 유지한 전체 비-중복 열 순서."""
    df = pd.read_excel(path, sheet_name=SHEET_METHODS, header=1)
    if "Unnamed: 15" in df.columns:
        df = df.rename(columns={"Unnamed: 15": "is_same"})
    entity_cols = [
        "source_id",
        "domain_id",
        "source_value",
        "gt_concept_id",
        "gt_concept_name",
    ]
    hub_cols = [c for c in df.columns if c.startswith("omophub_")]
    usagi_cols = [c for c in df.columns if c.startswith("usagi_")]
    phj_cols = [c for c in df.columns if str(c).startswith("PHJ")]
    tail = [c for c in ("is_same", "baseline") if c in df.columns]
    seen: set[str] = set()
    ordered: list[str] = []
    for group in (entity_cols, hub_cols, phj_cols, usagi_cols, tail):
        for c in group:
            if c in df.columns and c not in seen:
                ordered.append(c)
                seen.add(c)
    for c in df.columns:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return df[ordered].copy()


def load_human_df(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=SHEET_HUMAN)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    ap = argparse.ArgumentParser(description="SNUH drug baseline xlsx → methods+human 2 files")
    ap.add_argument(
        "-i",
        "--input",
        type=Path,
        default=root / "eval" / "SNUH_drug_평가자P_comparison_baseline.xlsx",
    )
    ap.add_argument("-o", "--output-dir", type=Path, default=root / "eval" / "out")
    args = ap.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        print(f"입력 없음: {inp}", file=sys.stderr)
        return 1

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out1 = out_dir / "SNUH_drug_entity_baseline_hub_usagi_phj.xlsx"
    out2 = out_dir / "SNUH_drug_evaluators_P_Y_A_final.xlsx"

    mdf_lo = load_methods_df_simple(inp)
    hdf = load_human_df(inp)

    with pd.ExcelWriter(out1, engine="openpyxl") as w:
        mdf_lo.to_excel(w, index=False, sheet_name="methods")
    with pd.ExcelWriter(out2, engine="openpyxl") as w:
        hdf.to_excel(w, index=False, sheet_name="human_scores")

    print(f"저장: {out1} ({len(mdf_lo)}행)")
    print(f"저장: {out2} ({len(hdf)}행)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
