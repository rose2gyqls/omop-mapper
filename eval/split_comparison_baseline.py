#!/usr/bin/env python3
"""
comparison_baseline + 평가점수 시트가 있는 통합 xlsx → 2개 산출 (PHJ 유지).

출력 이름: <입력파일stem>_entity_baseline_hub_usagi_phj.xlsx,
          <입력파일stem>_evaluators_P_Y_A_final.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SHEET_METHODS = "comparison_baseline"
SHEET_HUMAN = "평가점수_일치작업_평가자P만다름"


def load_methods_df_simple(path: Path) -> pd.DataFrame:
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


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="통합 baseline xlsx → methods + human 2 files")
    ap.add_argument("-i", "--input", type=Path, required=True)
    ap.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=root / "eval" / "out",
    )
    args = ap.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        print(f"입력 없음: {inp}", file=sys.stderr)
        return 1

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = inp.stem
    out1 = out_dir / f"{stem}_entity_baseline_hub_usagi_phj.xlsx"
    out2 = out_dir / f"{stem}_evaluators_P_Y_A_final.xlsx"

    mdf = load_methods_df_simple(inp)
    hdf = pd.read_excel(inp, sheet_name=SHEET_HUMAN)

    with pd.ExcelWriter(out1, engine="openpyxl") as w:
        mdf.to_excel(w, index=False, sheet_name="methods")
    with pd.ExcelWriter(out2, engine="openpyxl") as w:
        hdf.to_excel(w, index=False, sheet_name="human_scores")

    print(f"저장: {out1} ({len(mdf)}행)")
    print(f"저장: {out2} ({len(hdf)}행)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
