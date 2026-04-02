"""
OMOPHub 배치 출력 CSV → 요약 엑셀 (열 순서 고정).

열: source_id, domain_id, source_value,
    ground_truth_concept_id(선택), ground_truth_concept_name(선택),
    result_concept_id, result_concept_name, result_domain_id, results_score
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from omophub.top_hits import domain_id_from_top_hits_json

OUTPUT_COLUMNS = [
    "source_id",
    "domain_id",
    "source_value",
    "ground_truth_concept_id",
    "ground_truth_concept_name",
    "result_concept_id",
    "result_concept_name",
    "result_domain_id",
    "results_score",
]


def _nullable_int(val: Any) -> Any:
    if pd.isna(val) or val is None or (isinstance(val, str) and not str(val).strip()):
        return pd.NA
    try:
        x = float(val)
        if pd.isna(x):
            return pd.NA
        if abs(x - round(x)) < 1e-9:
            return int(round(x))
        return x
    except (TypeError, ValueError):
        return val


def omophub_csv_to_summary(df: pd.DataFrame) -> pd.DataFrame:
    """OMOPHub 출력 스키마 → 요약 열."""
    out = pd.DataFrame()
    out["source_id"] = df["source_id"] if "source_id" in df.columns else pd.NA
    out["domain_id"] = df["domain_id"] if "domain_id" in df.columns else pd.NA
    out["source_value"] = df["source_value"] if "source_value" in df.columns else pd.NA

    if "ground_truth_concept_id" in df.columns:
        out["ground_truth_concept_id"] = df["ground_truth_concept_id"].map(_nullable_int)
    else:
        out["ground_truth_concept_id"] = pd.NA

    if "ground_truth_concept_name" in df.columns:
        out["ground_truth_concept_name"] = df["ground_truth_concept_name"]
    else:
        out["ground_truth_concept_name"] = pd.NA

    if "result1_concept_id" in df.columns:
        out["result_concept_id"] = df["result1_concept_id"].map(_nullable_int)
    else:
        out["result_concept_id"] = pd.NA

    if "result1_concept_name" in df.columns:
        out["result_concept_name"] = df["result1_concept_name"]
    else:
        out["result_concept_name"] = pd.NA

    # 결과 domain은 top_hits_json 첫 히트 기준(입력 domain과 무관)
    if "top_hits_json" in df.columns:
        out["result_domain_id"] = df["top_hits_json"].map(domain_id_from_top_hits_json)
    elif "result1_domain_id" in df.columns:
        out["result_domain_id"] = df["result1_domain_id"]
    else:
        out["result_domain_id"] = pd.NA

    if "result1_score" in df.columns:
        out["results_score"] = df["result1_score"]
    else:
        out["results_score"] = pd.NA

    return out[OUTPUT_COLUMNS]


def _latest_match(directory: Path, pattern: str) -> Path | None:
    paths = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0] if paths else None


def export_excel(
    csv_path: Path,
    xlsx_path: Path,
) -> Path:
    df = pd.read_csv(csv_path)
    summary = omophub_csv_to_summary(df)
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_excel(xlsx_path, index=False, engine="openpyxl")
    return xlsx_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OMOPHub 출력 CSV를 요약 엑셀로 저장합니다."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_root / "omophub" / "outputs",
        help="기본: 프로젝트 omophub/outputs",
    )
    parser.add_argument(
        "--snomed-csv",
        type=Path,
        default=None,
        help="SNOMED용 입력 CSV (미지정 시 input-dir 에서 *snomed* 최신 파일)",
    )
    parser.add_argument(
        "--snuh-csv",
        type=Path,
        default=None,
        help="SNUH용 입력 CSV (미지정 시 input-dir 에서 *snuh* 또는 *baseline* 최신 파일)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="엑셀 저장 디렉터리 (기본: input-dir)",
    )
    args = parser.parse_args()

    indir = args.input_dir.resolve()
    outdir = (args.out_dir or indir).resolve()

    snomed = args.snomed_csv
    if snomed is None:
        p = _latest_match(indir, "*snomed*.csv")
        snomed = p
    else:
        snomed = snomed.resolve()

    snuh = args.snuh_csv
    if snuh is None:
        p = _latest_match(indir, "*snuh*.csv")
        if p is None:
            p = _latest_match(indir, "*baseline*.csv")
        snuh = p
    else:
        snuh = snuh.resolve()

    if snomed is None or not snomed.is_file():
        print("오류: SNOMED용 CSV를 찾지 못했습니다. --snomed-csv 로 지정하세요.", file=sys.stderr)
        sys.exit(1)
    if snuh is None or not snuh.is_file():
        print("오류: SNUH용 CSV를 찾지 못했습니다. --snuh-csv 로 지정하세요.", file=sys.stderr)
        sys.exit(1)

    snomed_xlsx = outdir / "omophub_snomed_summary.xlsx"
    snuh_xlsx = outdir / "omophub_snuh_summary.xlsx"

    export_excel(snomed, snomed_xlsx)
    export_excel(snuh, snuh_xlsx)
    print(f"SNOMED: {snomed_xlsx}")
    print(f"SNUH:   {snuh_xlsx}")


if __name__ == "__main__":
    main()
