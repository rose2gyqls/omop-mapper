#!/usr/bin/env python3
"""
981건 Drug × 20회 run × 평가자(P,Y,A) + 최종 종합 평가 시트.

- quantitative_evaluation_snuh_meas.xlsx: P/Y/A_snuh_drug, 평가상이_목록(Drug)
- SNUH_drug_평가자P_comparison_baseline.xlsx: 평가점수 시트 「최종」합의 →
  (entity, concept) 일치 run에서 P/Y/A 동일 점수
- mapping_snuh_drug_merged.json: run별 best_concept_id

Drug 4단계 점수: Acc_2(%), Acc_2+1(%), 가중평균(%) — compute_quantitative_metrics.py 와 동일.

Usage:
  python eval/build_snuh_drug_run_matrix.py \\
    --json test_logs/SNUH/mapping_snuh_drug_merged.json \\
    --quant-xlsx eval/quantitative_evaluation_snuh_meas.xlsx \\
    --baseline-xlsx eval/SNUH_drug_평가자P_comparison_baseline.xlsx \\
    -o eval/out/SNUH_drug_981x20_evaluators.xlsx
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_eval_dir = Path(__file__).resolve().parent
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

import pandas as pd

from consensus_run20_wide import build_consensus_wide_human_map

NUM_ROWS = 981
NUM_RUNS = 20
EVALUATORS = ("P", "Y", "A")
SHEET_DISPUTE = "평가상이_목록"
SHEET_CONSENSUS = "평가점수_일치작업_평가자P만다름"
SHEET_METHODS = "comparison_baseline"
QUANT_SHEETS = {"P": "P_snuh_drug", "Y": "Y_snuh_drug", "A": "A_snuh_drug"}
DOMAIN_KEY = "drug"

SNUH_DOMAINS = ("Condition", "Measurement", "Procedure", "Drug")
SNOMED_DOMAINS = ("Condition", "Measurement", "Procedure", "Observation")


def _parse_trailing_concept_id(concept_cell) -> int | None:
    if concept_cell is None or (isinstance(concept_cell, float) and pd.isna(concept_cell)):
        return None
    m = re.search(r"\((\d+)\)\s*$", str(concept_cell).strip())
    return int(m.group(1)) if m else None


def load_consensus_final(baseline_xlsx: Path) -> dict[tuple[str, int], float]:
    df = pd.read_excel(baseline_xlsx, sheet_name=SHEET_CONSENSUS)
    need = {"entity_name", "concept", "최종"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"{SHEET_CONSENSUS}: 컬럼 필요 {need}, 실제 {list(df.columns)}")
    out: dict[tuple[str, int], float] = {}
    for _, row in df.iterrows():
        ek = str(row["entity_name"]).strip().casefold()
        cid = _parse_trailing_concept_id(row.get("concept"))
        if cid is None:
            continue
        v = row.get("최종")
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        out[(ek, cid)] = float(v)
    return out


def load_dispute_drug(quant_xlsx: Path) -> dict[tuple[str, int], dict[str, float | None]]:
    df = pd.read_excel(quant_xlsx, sheet_name=SHEET_DISPUTE)
    dom = df["domain_id"].astype(str).str.strip().str.casefold()
    df = df.loc[dom == DOMAIN_KEY].copy()
    out: dict[tuple[str, int], dict[str, float | None]] = {}
    for _, row in df.iterrows():
        ek = str(row["entity_name"]).strip().casefold()
        cid = _parse_trailing_concept_id(row.get("concept"))
        if cid is None:
            continue

        def _f(col: str) -> float | None:
            v = row.get(col)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            return float(v)

        out[(ek, cid)] = {"P": _f("P"), "Y": _f("Y"), "A": _f("A")}
    return out


def load_quant_run_tables(quant_xlsx: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    P = pd.read_excel(quant_xlsx, sheet_name=QUANT_SHEETS["P"])
    Y = pd.read_excel(quant_xlsx, sheet_name=QUANT_SHEETS["Y"])
    A = pd.read_excel(quant_xlsx, sheet_name=QUANT_SHEETS["A"])
    for name, d in ("P", P), ("Y", Y), ("A", A):
        if len(d) != NUM_ROWS:
            raise ValueError(f"{QUANT_SHEETS[name]}: {NUM_ROWS}행 필요, 실제 {len(d)}")
        for r in range(1, NUM_RUNS + 1):
            if f"run{r}" not in d.columns:
                raise ValueError(f"{QUANT_SHEETS[name]}: run{r} 열 없음")
    return P, Y, A


def load_optional_source_ids(baseline_xlsx: Path | None) -> list | None:
    if baseline_xlsx is None or not baseline_xlsx.is_file():
        return None
    df = pd.read_excel(baseline_xlsx, sheet_name=SHEET_METHODS, header=1)
    if "source_id" not in df.columns or len(df) != NUM_ROWS:
        return None
    return [df.iloc[i]["source_id"] for i in range(NUM_ROWS)]


def load_json_runs(json_path: Path) -> list[list[dict]]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    runs = data.get("runs")
    if not runs or len(runs) != NUM_RUNS:
        raise ValueError(f"JSON runs 기대 {NUM_RUNS}, 실제 {len(runs) if runs else 0}")
    for r, run in enumerate(runs):
        if len(run) != NUM_ROWS:
            raise ValueError(f"run {r+1}: 행 수 기대 {NUM_ROWS}, 실제 {len(run)}")
    return runs


def _parse_best_concept_id(raw) -> int | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("none", "nan"):
        return None
    return int(s)


def best_concept_matrix(runs: list[list[dict]]) -> list[list[int | None]]:
    mat: list[list[int | None]] = []
    for run in runs:
        by_id: dict[int, int | None] = {}
        for x in run:
            rid = int(x["row_id"])
            by_id[rid] = _parse_best_concept_id(x.get("best_concept_id"))
        mat.append([by_id.get(i) for i in range(NUM_ROWS)])
    return mat


def _cell_changed(old_v, new_v: float) -> bool:
    if old_v is pd.NA or (isinstance(old_v, (int, float)) and pd.isna(old_v)):
        return True
    try:
        return float(old_v) != float(new_v)
    except (TypeError, ValueError):
        return True


def build_matrix(
    P: pd.DataFrame,
    Y: pd.DataFrame,
    A: pd.DataFrame,
    concept_mat: list[list[int | None]],
    consensus: dict[tuple[str, int], float],
    dispute: dict[tuple[str, int], dict[str, float | None]],
    source_ids: list | None,
) -> tuple[pd.DataFrame, int, int, int, int]:
    rows: list[dict] = []
    n_cons_apply = n_cons_changed = n_dis_apply = n_dis_changed = 0

    for i in range(NUM_ROWS):
        entity = str(P.iloc[i]["entity_name"]).strip()
        ek = entity.casefold()
        base: dict = {
            "index": i + 1,
            "entity_name": entity,
            "domain_id": P.iloc[i]["domain"],
        }
        if source_ids is not None:
            base["source_id"] = source_ids[i]

        for ev, sheet in ("P", P), ("Y", Y), ("A", A):
            for r in range(NUM_RUNS):
                col = f"{ev}_run{r + 1:02d}"
                v = sheet.iloc[i][f"run{r + 1}"]
                base[col] = v if pd.notna(v) else pd.NA

        for r in range(NUM_RUNS):
            cid = concept_mat[r][i]
            if cid is None:
                continue
            key = (ek, cid)
            if key in consensus:
                fv = consensus[key]
                for ev in EVALUATORS:
                    col = f"{ev}_run{r + 1:02d}"
                    old_v = base[col]
                    n_cons_apply += 1
                    if _cell_changed(old_v, fv):
                        n_cons_changed += 1
                    base[col] = fv
                continue
            if key not in dispute:
                continue
            adj = dispute[key]
            for ev in EVALUATORS:
                col = f"{ev}_run{r + 1:02d}"
                if adj.get(ev) is None:
                    continue
                new_v = adj[ev]
                old_v = base[col]
                n_dis_apply += 1
                if _cell_changed(old_v, new_v):
                    n_dis_changed += 1
                base[col] = new_v

        rows.append(base)

    return pd.DataFrame(rows), n_cons_apply, n_cons_changed, n_dis_apply, n_dis_changed


def build_concept_audit(concept_mat: list[list[int | None]]) -> pd.DataFrame:
    data = {"index": list(range(1, NUM_ROWS + 1))}
    for r in range(NUM_RUNS):
        data[f"best_concept_id_run{r + 1:02d}"] = [concept_mat[r][i] for i in range(NUM_ROWS)]
    return pd.DataFrame(data)


def _drug_acc_metrics(flat_scores: list[float]) -> tuple[float, float, float]:
    """Acc_2 %, Acc_2+1 %, 가중평균(0~100, 평균)."""
    if not flat_scores:
        return float("nan"), float("nan"), float("nan")
    n = len(flat_scores)
    acc_2 = sum(1.0 for s in flat_scores if s >= 2.0) / n * 100.0
    acc_2_1 = sum(1.0 for s in flat_scores if s >= 1.0) / n * 100.0
    weighted = (
        sum(
            100.0 if s >= 2.0 else (50.0 if s >= 1.0 else (25.0 if s >= 0.5 else 0.0))
            for s in flat_scores
        )
        / n
    )
    return acc_2, acc_2_1, weighted


def _drug_evaluator_metrics(df_scores: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    for ev in EVALUATORS:
        flat: list[float] = []
        for r in range(NUM_RUNS):
            c = f"{ev}_run{r + 1:02d}"
            for v in df_scores[c]:
                if pd.notna(v):
                    flat.append(float(v))
        out[ev] = _drug_acc_metrics(flat)
    return out


def build_summary_final_sheet(df_scores: pd.DataFrame) -> pd.DataFrame:
    """Drug 도메인 행만 수치 채움, 나머지 도메인은 비움."""
    m = _drug_evaluator_metrics(df_scores)
    dom_acc = sum(m[ev][0] for ev in EVALUATORS) / 3.0
    filled_label = "Drug"

    rows: list[dict[str, object]] = []

    def _block(dataset: str, domains: tuple[str, ...], first_row_in_dataset: bool) -> None:
        nonlocal rows
        for di, domain in enumerate(domains):
            for ei, ev in enumerate(EVALUATORS):
                is_first_row = di == 0 and ei == 0 and first_row_in_dataset
                is_first_in_domain = ei == 0

                acc2 = pd.NA
                wavg = pd.NA
                acc21 = pd.NA
                dom_cell = pd.NA
                total_cell = pd.NA

                if di == 0 and ei == 0 and dataset == "SNUH":
                    total_cell = round(dom_acc, 2)

                if domain == filled_label:
                    a2, a21, w = m[ev]
                    acc2 = round(a2, 2)
                    acc21 = round(a21, 2)
                    wavg = round(w, 2)
                    if is_first_in_domain:
                        dom_cell = round(dom_acc, 2)

                rows.append(
                    {
                        "데이터": dataset if is_first_row else pd.NA,
                        "도메인": domain if is_first_in_domain else pd.NA,
                        "평가자": ev,
                        "Acc_2(%)": acc2,
                        "Acc_2+1(%, drug)": acc21,
                        "가중평균(%)": wavg,
                        "도메인 정확도": dom_cell,
                        "총 정확도": total_cell,
                    }
                )

    _block("SNUH", SNUH_DOMAINS, True)
    _block("SNOMED", SNOMED_DOMAINS, True)
    assert len(rows) == 24
    return pd.DataFrame(rows)


def build_single_evaluator_run_sheet(df_scores: pd.DataFrame, ev: str) -> pd.DataFrame:
    base_cols = ["index", "entity_name"]
    if "source_id" in df_scores.columns:
        base_cols.insert(1, "source_id")
    part = df_scores[base_cols + ["domain_id"]].copy().rename(columns={"domain_id": "domain"})
    rename_runs = {f"{ev}_run{r + 1:02d}": f"run{r + 1}" for r in range(NUM_RUNS)}
    runs = df_scores[[f"{ev}_run{r + 1:02d}" for r in range(NUM_RUNS)]].rename(columns=rename_runs)
    return pd.concat([part.reset_index(drop=True), runs.reset_index(drop=True)], axis=1)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    ap = argparse.ArgumentParser(description="SNUH Drug 981×20 × P/Y/A")
    ap.add_argument(
        "--json",
        type=Path,
        default=root / "test_logs" / "SNUH" / "mapping_snuh_drug_merged.json",
    )
    ap.add_argument(
        "--quant-xlsx",
        type=Path,
        default=root / "eval" / "quantitative_evaluation_snuh_meas.xlsx",
    )
    ap.add_argument(
        "--baseline-xlsx",
        type=Path,
        default=root / "eval" / "SNUH_drug_평가자P_comparison_baseline.xlsx",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=root / "eval" / "out" / "SNUH_drug_981x20_evaluators.xlsx",
    )
    args = ap.parse_args()

    jpath = args.json.expanduser().resolve()
    qpath = args.quant_xlsx.expanduser().resolve()
    out = args.output.expanduser().resolve()
    baseline = args.baseline_xlsx.expanduser().resolve()

    if not jpath.is_file():
        print(f"JSON 없음: {jpath}", file=sys.stderr)
        return 1
    if not qpath.is_file():
        print(f"quant xlsx 없음: {qpath}", file=sys.stderr)
        return 1

    P, Y, A = load_quant_run_tables(qpath)
    dispute = load_dispute_drug(qpath)
    runs = load_json_runs(jpath)
    concept_mat = best_concept_matrix(runs)

    consensus: dict[tuple[str, int], float] = {}
    source_ids = None
    if baseline.is_file():
        consensus = load_consensus_final(baseline)
        source_ids = load_optional_source_ids(baseline)
        print(f"합의(최종) 키: {len(consensus)} ({baseline.name})")
    else:
        print("경고: baseline 없음 — 합의 생략", file=sys.stderr)

    df_scores, nc_app, nc_ch, nd_app, nd_ch = build_matrix(
        P, Y, A, concept_mat, consensus, dispute, source_ids
    )
    df_audit = build_concept_audit(concept_mat)
    df_summary = build_summary_final_sheet(df_scores)
    entities = [str(P.iloc[i]["entity_name"]).strip() for i in range(NUM_ROWS)]
    domain_col = str(P.iloc[0]["domain"]).strip()

    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_summary.to_excel(writer, index=False, sheet_name="최종 종합 평가")
        build_consensus_wide_human_map(
            entities=entities,
            domain=domain_col,
            concept_mat=concept_mat,
            human_scores=consensus,
            source_ids=source_ids,
            df_scores=df_scores,
        ).to_excel(writer, index=False, sheet_name="consensus_run20")
        build_single_evaluator_run_sheet(df_scores, "P").to_excel(
            writer, index=False, sheet_name="P_drug"
        )
        build_single_evaluator_run_sheet(df_scores, "Y").to_excel(
            writer, index=False, sheet_name="Y_drug"
        )
        build_single_evaluator_run_sheet(df_scores, "A").to_excel(
            writer, index=False, sheet_name="A_drug"
        )
        df_audit.to_excel(writer, index=False, sheet_name="best_concept_per_run")

    score_cols = [c for c in df_scores.columns if re.match(r"^[PYA]_run\d{2}$", c)]
    filled = df_scores[score_cols].notna().sum().sum()
    total = len(df_scores) * len(score_cols)
    print(f"저장: {out}")
    print(f"점수 칸: {filled}/{total} ({100.0 * filled / total:.1f}%)")
    print(f"합의 적용 칸: {nc_app} (변경: {nc_ch})")
    print(f"평가상이 적용: {nd_app} (변경: {nd_ch}), 키 {len(dispute)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
