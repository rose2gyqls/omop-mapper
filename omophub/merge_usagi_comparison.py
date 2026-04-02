"""
OMOPHub 요약 엑셀 + USAGI 평가 CSV 를 sourceName·domain_id 기준으로 병합합니다.

기본 동작: 원본 입력 CSV(data/snomed…, data/snuh…) 행 순서를 유지한 채
(domain_id + 텍스트) 키로 OMOPHub·USAGI 결과를 left join 합니다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import pandas as pd

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

USAGI_DOMAIN_COL = "ADD_INFO:domain_id"
USAGI_TEXT_COL = "sourceName"

OMOPHUB_RENAME = {
    "result_concept_id": "omophub_result_concept_id",
    "result_concept_name": "omophub_result_concept_name",
    "results_score": "omophub_results_score",
}

USAGI_RENAME = {
    "targetConceptId": "usagi_targetConceptId",
    "targetConceptName": "usagi_targetConceptName",
    "targetDomainId": "usagi_targetDomainId",
    "targetStandardConcept": "usagi_targetStandardConcept",
    "matchScore": "usagi_matchScore",
}

FINAL_COLUMN_ORDER = [
    "source_id",
    "domain_id",
    "source_value",
    "ground_truth_concept_id",
    "ground_truth_concept_name",
    "omophub_result_concept_id",
    "omophub_result_concept_name",
    "omophub_results_score",
    "usagi_targetConceptId",
    "usagi_targetConceptName",
    "usagi_targetDomainId",
    "usagi_targetStandardConcept",
    "usagi_matchScore",
]


def _strip_key(val) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip()


def _add_merge_keys(df: pd.DataFrame, domain_col: str, text_col: str) -> pd.DataFrame:
    out = df.copy()
    out["__k_domain"] = out[domain_col].map(_strip_key)
    out["__k_text"] = out[text_col].map(_strip_key)
    return out


def load_usagi_concat(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p, encoding="utf-8") for p in paths]
    usa_raw = pd.concat(frames, ignore_index=True)
    cols_keep = [
        USAGI_DOMAIN_COL,
        USAGI_TEXT_COL,
        "targetConceptId",
        "targetConceptName",
        "targetDomainId",
        "targetStandardConcept",
        "matchScore",
    ]
    for c in cols_keep:
        if c not in usa_raw.columns:
            raise ValueError(f"USAGI CSV에 열 '{c}' 가 없습니다: {paths}")
    usa_raw = usa_raw[cols_keep].copy()
    usa_raw = _add_merge_keys(usa_raw, USAGI_DOMAIN_COL, USAGI_TEXT_COL)
    usa_raw = usa_raw.drop_duplicates(subset=["__k_domain", "__k_text"], keep="first")
    usa_raw = usa_raw.rename(columns=USAGI_RENAME)
    return usa_raw


def _master_from_input(
    input_csv: Path,
    dataset: Literal["snomed", "snuh"],
) -> pd.DataFrame:
    """원본 입력 CSV → source_id, domain_id, source_value, ground_truth_* (입력 순서 유지)."""
    raw = pd.read_csv(input_csv)
    if dataset == "snomed":
        master = pd.DataFrame(
            {
                "source_id": raw["test_index"],
                "domain_id": raw["domain_id"],
                "source_value": raw["entity_name"],
            }
        )
        if "concept_id" in raw.columns:
            master["ground_truth_concept_id"] = raw["concept_id"]
        else:
            master["ground_truth_concept_id"] = pd.NA
        if "concept_name" in raw.columns:
            master["ground_truth_concept_name"] = raw["concept_name"]
        else:
            master["ground_truth_concept_name"] = pd.NA
    else:
        master = pd.DataFrame(
            {
                "source_id": raw["no"],
                "domain_id": raw["domain_id"],
                "source_value": raw["source_value"],
                "ground_truth_concept_id": raw["concept_id"],
                "ground_truth_concept_name": raw["concept_name"],
            }
        )
    return master


def merge_input_ordered(
    input_csv: Path,
    omophub_xlsx: Path,
    usagi_csvs: list[Path],
    dataset: Literal["snomed", "snuh"],
) -> tuple[pd.DataFrame, str]:
    """입력 CSV 행 순서 = 출력 행 순서. OMOPHub·USAGI는 (domain_id, source_value) 키로 left join."""
    master = _master_from_input(input_csv, dataset)
    master = _add_merge_keys(master, "domain_id", "source_value")

    omo = pd.read_excel(omophub_xlsx, engine="openpyxl")
    omo = omo.rename(columns=OMOPHUB_RENAME)
    omo = _add_merge_keys(omo, "domain_id", "source_value")
    dup_omo = int(omo.duplicated(subset=["__k_domain", "__k_text"]).sum())
    dup_note = ""
    if dup_omo:
        omo = omo.drop_duplicates(subset=["__k_domain", "__k_text"], keep="first")
        dup_note = f"OMOPHub 요약에서 동일 (domain_id, source_value) 키 {dup_omo}건 제거 후 조인(첫 행 유지)"

    omo_cols = [
        "__k_domain",
        "__k_text",
        "omophub_result_concept_id",
        "omophub_result_concept_name",
        "omophub_results_score",
    ]
    omo = omo[[c for c in omo_cols if c in omo.columns]]

    m = master.merge(omo, on=["__k_domain", "__k_text"], how="left")

    usa = load_usagi_concat(usagi_csvs)
    usa_cols = ["__k_domain", "__k_text"] + list(USAGI_RENAME.values())
    usa = usa[[c for c in usa_cols if c in usa.columns]]

    m = m.merge(usa, on=["__k_domain", "__k_text"], how="left")
    m = m.drop(columns=["__k_domain", "__k_text"], errors="ignore")

    for c in FINAL_COLUMN_ORDER:
        if c not in m.columns:
            m[c] = pd.NA
    m = m[FINAL_COLUMN_ORDER]
    return m, dup_note


def _nullable_int_series(s: pd.Series) -> pd.Series:
    def one(v):
        if pd.isna(v) or v is None or (isinstance(v, str) and not str(v).strip()):
            return pd.NA
        try:
            x = float(v)
            if pd.isna(x):
                return pd.NA
            if abs(x - round(x)) < 1e-9:
                return int(round(x))
            return x
        except (TypeError, ValueError):
            return v

    return s.map(one)


def polish_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in (
        "ground_truth_concept_id",
        "omophub_result_concept_id",
        "usagi_targetConceptId",
    ):
        if col in out.columns:
            out[col] = _nullable_int_series(out[col])
    return out


def run(
    input_csv: Path,
    omophub_xlsx: Path,
    usagi_csvs: list[Path],
    out_xlsx: Path,
    dataset: Literal["snomed", "snuh"],
    *,
    verbose: bool = True,
) -> Path:
    merged, dup_note = merge_input_ordered(
        input_csv, omophub_xlsx, usagi_csvs, dataset
    )
    if verbose and dup_note:
        print(f"  [{omophub_xlsx.name}] {dup_note}")
    merged = polish_dtypes(merged)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    merged.to_excel(out_xlsx, index=False, engine="openpyxl")
    return out_xlsx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="입력 CSV 순서 기준 OMOPHub 요약 xlsx + USAGI CSV 병합 (domain + 텍스트 키)"
    )
    out_dir = _root / "omophub" / "outputs"
    data_dir = _root / "data"
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=out_dir,
        help="저장 디렉터리",
    )
    parser.add_argument(
        "--snomed-input",
        type=Path,
        default=data_dir / "snomed-mapping-data-1000.csv",
        help="SNOMED 입력(행 순서 기준)",
    )
    parser.add_argument(
        "--snuh-input",
        type=Path,
        default=data_dir / "snuh-baseline-mapping-data.csv",
        help="SNUH 입력(행 순서 기준)",
    )
    parser.add_argument(
        "--snomed-omophub",
        type=Path,
        default=out_dir / "omophub_snomed_summary.xlsx",
    )
    parser.add_argument(
        "--snuh-omophub",
        type=Path,
        default=out_dir / "omophub_snuh_summary.xlsx",
    )
    parser.add_argument(
        "--usagi-snomed",
        type=Path,
        default=out_dir / "evaluation_USAGI_test_3_SNOMED_용어.xlsx.csv",
    )
    parser.add_argument(
        "--usagi-snuh-condition",
        type=Path,
        default=out_dir / "evaluation_USAGI_test_1_SNUH_condition_용어.xlsx.csv",
    )
    parser.add_argument(
        "--usagi-snuh-rest",
        type=Path,
        default=out_dir / "evaluation_USAGI_test_2_SNUH_drug_meas_proc_용어.xlsx.csv",
    )
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()

    p1 = run(
        args.snomed_input.resolve(),
        args.snomed_omophub.resolve(),
        [args.usagi_snomed.resolve()],
        out_dir / "comparison_SNOMED_omophub_usagi.xlsx",
        "snomed",
        verbose=True,
    )
    p2 = run(
        args.snuh_input.resolve(),
        args.snuh_omophub.resolve(),
        [args.usagi_snuh_condition.resolve(), args.usagi_snuh_rest.resolve()],
        out_dir / "comparison_SNUH_omophub_usagi.xlsx",
        "snuh",
        verbose=True,
    )
    print(f"SNOMED 병합 ({args.snomed_input.name} 순서): {p1}")
    print(f"SNUH 병합 ({args.snuh_input.name} 순서): {p2}")


if __name__ == "__main__":
    main()
