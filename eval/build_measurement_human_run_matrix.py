#!/usr/bin/env python3
"""
110건 Measurement × 20회 run × 평가자(P,Y,A) 점수 매트릭스 생성.

데이터 출처 (eval/quantitative_evaluation_snuh_meas.xlsx):
  - P/Y/A_snuh_measurement: 평가자별 run1..run20 기본 점수.

eval/SNUH_measurement_평가자P_comparison_baseline.xlsx — 시트 평가점수_일치작업_평가자P만다름:
  - 열 「최종」은 3인 합의 점수. (entity_name, concept_id)가 JSON run의 best_concept_id와
    일치하면 그 run의 P_run/Y_run/A_run 을 **모두 최종 값으로 통일**한다.
  - 그 외 (entity, concept) 조합은 평가자별 quant 점수 유지.

평가상이_목록(quant 내): 위 합의 키와 겹치면 무시(합의 우선). 그 외에만 P,Y,A 개별 조정 적용.

검증용 JSON: run별 best_concept_id (mapping_snuh_*_meas.json).

최종 종합 평가의 Acc_2·가중평균·도메인/총 정확도는 **위 규칙으로 합친 뒤** 110×20×3 셀에서 산출한다.

출력 시트: 최종 종합 평가 + P_measurement / Y_measurement / A_measurement + best_concept_per_run.

Usage:
  python eval/build_measurement_human_run_matrix.py \\
    --json test_logs/SNUH/mapping_snuh_20260313_meas.json \\
    --quant-xlsx eval/quantitative_evaluation_snuh_meas.xlsx \\
    --baseline-xlsx eval/SNUH_measurement_평가자P_comparison_baseline.xlsx \\
    -o eval/out/SNUH_measurement_110x20_evaluators.xlsx
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

SHEET_DISPUTE = "평가상이_목록"
SHEET_CONSENSUS = "평가점수_일치작업_평가자P만다름"
SHEET_METHODS = "comparison_baseline"
NUM_RUNS = 20
EVALUATORS = ("P", "Y", "A")
QUANT_SHEETS = {"P": "P_snuh_measurement", "Y": "Y_snuh_measurement", "A": "A_snuh_measurement"}

# 최종 종합 평가 시트: SNUH 4도메인 + SNOMED 4도메인 (원본 quantitative 템플릿과 동일 순서)
SNUH_DOMAINS = ("Condition", "Measurement", "Procedure", "Drug")
SNOMED_DOMAINS = ("Condition", "Measurement", "Procedure", "Observation")


def _parse_trailing_concept_id(concept_cell) -> int | None:
    if concept_cell is None or (isinstance(concept_cell, float) and pd.isna(concept_cell)):
        return None
    m = re.search(r"\((\d+)\)\s*$", str(concept_cell).strip())
    return int(m.group(1)) if m else None


def load_consensus_final(baseline_xlsx: Path) -> dict[tuple[str, int], float]:
    """평가점수 시트: (entity casefold, concept_id) -> 3인 합의 「최종」 점수."""
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


def load_dispute_overrides(quant_xlsx: Path) -> dict[tuple[str, int], dict[str, float | None]]:
    """평가상이_목록 Measurement: (entity casefold, concept_id) -> 조정된 P,Y,A."""
    df = pd.read_excel(quant_xlsx, sheet_name=SHEET_DISPUTE)
    dom = df["domain_id"].astype(str).str.strip().str.casefold()
    df = df.loc[dom == "measurement"].copy()
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
        if len(d) != 110:
            raise ValueError(f"{QUANT_SHEETS[name]}: 110행 필요, 실제 {len(d)}")
        for r in range(1, NUM_RUNS + 1):
            if f"run{r}" not in d.columns:
                raise ValueError(f"{QUANT_SHEETS[name]}: run{r} 열 없음")
    return P, Y, A


def load_optional_source_ids(baseline_xlsx: Path | None) -> list | None:
    if baseline_xlsx is None or not baseline_xlsx.is_file():
        return None
    df = pd.read_excel(baseline_xlsx, sheet_name=SHEET_METHODS, header=1)
    if "source_id" not in df.columns or len(df) != 110:
        return None
    return [df.iloc[i]["source_id"] for i in range(110)]


def load_json_runs(json_path: Path) -> list[list[dict]]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    runs = data.get("runs")
    if not runs or len(runs) != NUM_RUNS:
        raise ValueError(f"JSON runs 개수 기대 {NUM_RUNS}, 실제 {len(runs) if runs else 0}")
    for r, run in enumerate(runs):
        if len(run) != 110:
            raise ValueError(f"run {r+1}: 행 수 기대 110, 실제 {len(run)}")
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
        mat.append([by_id.get(i) for i in range(110)])
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
    """quant → (합의 최종이 있으면 P/Y/A 동일값) → (평가상이, 합의 키 제외).

    반환: (df, 합의 적용 칸 수, 합의로 값 바뀐 칸 수, 평가상이 적용 칸, 평가상이로 바뀐 칸)
    """
    rows: list[dict] = []
    n_cons_apply = 0
    n_cons_changed = 0
    n_dis_apply = 0
    n_dis_changed = 0

    for i in range(110):
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
                final_v = consensus[key]
                for ev in EVALUATORS:
                    col = f"{ev}_run{r + 1:02d}"
                    old_v = base[col]
                    n_cons_apply += 1
                    if _cell_changed(old_v, final_v):
                        n_cons_changed += 1
                    base[col] = final_v
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
    data = {"index": list(range(1, 111))}
    for r in range(NUM_RUNS):
        data[f"best_concept_id_run{r + 1:02d}"] = [concept_mat[r][i] for i in range(110)]
    return pd.DataFrame(data)


def _acc2_and_weighted(flat_scores: list[float]) -> tuple[float, float]:
    """Condition/Procedure/Measurement 3단계: 2→100%, 1→50%, 0→0% (compute_quantitative_metrics.py 와 동일)."""
    if not flat_scores:
        return float("nan"), float("nan")
    n = len(flat_scores)
    acc_2 = sum(1.0 for s in flat_scores if s >= 2.0) / n * 100.0
    weighted = (
        sum(1.0 if s >= 2.0 else (0.5 if s >= 1.0 else 0.0) for s in flat_scores) / n * 100.0
    )
    return acc_2, weighted


def _measurement_evaluator_metrics(df_scores: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """평가자별 (Acc_2%, 가중평균%) — run01~20 전체 셀(비어 있지 않은 값) 기준."""
    out: dict[str, tuple[float, float]] = {}
    for ev in EVALUATORS:
        cols = [f"{ev}_run{r + 1:02d}" for r in range(NUM_RUNS)]
        flat: list[float] = []
        for c in cols:
            for v in df_scores[c]:
                if pd.notna(v):
                    flat.append(float(v))
        out[ev] = _acc2_and_weighted(flat)
    return out


def build_summary_final_sheet(
    df_scores: pd.DataFrame,
    *,
    measurement_domain_label: str = "Measurement",
) -> pd.DataFrame:
    """
    원본 `최종 종합 평가` 시트와 동일한 열/행 구조.
    본 스크립트는 Measurement 매트릭스만 확실히 계산하고, 나머지 도메인·SNOMED는 비움.
    총 정확도(첫 행): Measurement 도메인 정확도와 동일하게 두어 단일 도메인 산출물의 대표 수치로 쓸 수 있음.
    """
    m = _measurement_evaluator_metrics(df_scores)
    dom_acc = sum(m[ev][0] for ev in EVALUATORS) / 3.0

    rows: list[dict[str, object]] = []

    def _block(
        dataset: str,
        domains: tuple[str, ...],
        first_row_in_dataset: bool,
    ) -> None:
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

                # 총 정확도: SNUH 표의 맨 윗줄(Condition·P) 한 칸 — Measurement만 계산된 경우 도메인 정확도와 동일
                if di == 0 and ei == 0 and dataset == "SNUH":
                    total_cell = round(dom_acc, 2)

                if domain == measurement_domain_label:
                    acc2 = round(m[ev][0], 2)
                    wavg = round(m[ev][1], 2)
                    if is_first_in_domain:
                        dom_cell = round(dom_acc, 2)

                row = {
                    "데이터": dataset if is_first_row else pd.NA,
                    "도메인": domain if is_first_in_domain else pd.NA,
                    "평가자": ev,
                    "Acc_2(%)": acc2,
                    "Acc_2+1(%, drug)": acc21,
                    "가중평균(%)": wavg,
                    "도메인 정확도": dom_cell,
                    "총 정확도": total_cell,
                }
                rows.append(row)

    _block("SNUH", SNUH_DOMAINS, True)
    _block("SNOMED", SNOMED_DOMAINS, True)
    assert len(rows) == 24
    return pd.DataFrame(rows)


def build_single_evaluator_run_sheet(df_scores: pd.DataFrame, ev: str) -> pd.DataFrame:
    """시트 P_measurement / Y_measurement / A_measurement — index, entity_name, domain, run1..run20."""
    base_cols = ["index", "entity_name"]
    if "source_id" in df_scores.columns:
        base_cols.insert(1, "source_id")
    use = base_cols + ["domain_id"]
    part = df_scores[use].copy()
    part = part.rename(columns={"domain_id": "domain"})
    rename_runs = {f"{ev}_run{r + 1:02d}": f"run{r + 1}" for r in range(NUM_RUNS)}
    runs = df_scores[[f"{ev}_run{r + 1:02d}" for r in range(NUM_RUNS)]].rename(columns=rename_runs)
    return pd.concat([part.reset_index(drop=True), runs.reset_index(drop=True)], axis=1)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    ap = argparse.ArgumentParser(
        description="Measurement 110×20 × P/Y/A (quant + 합의 최종 + 평가상이)"
    )
    ap.add_argument(
        "--json",
        type=Path,
        default=root / "test_logs" / "SNUH" / "mapping_snuh_20260313_meas.json",
    )
    ap.add_argument(
        "--quant-xlsx",
        type=Path,
        default=root / "eval" / "quantitative_evaluation_snuh_meas.xlsx",
    )
    ap.add_argument(
        "--baseline-xlsx",
        type=Path,
        default=root / "eval" / "SNUH_measurement_평가자P_comparison_baseline.xlsx",
        help="평가점수_일치작업_평가자P만다름(최종 합의)+comparison_baseline(source_id). 기본: eval/ 아래 파일",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=root / "eval" / "out" / "SNUH_measurement_110x20_evaluators.xlsx",
    )
    args = ap.parse_args()

    jpath = args.json.expanduser().resolve()
    qpath = args.quant_xlsx.expanduser().resolve()
    out = args.output.expanduser().resolve()
    baseline = args.baseline_xlsx.expanduser().resolve() if args.baseline_xlsx else None

    if not jpath.is_file():
        print(f"JSON 없음: {jpath}", file=sys.stderr)
        return 1
    if not qpath.is_file():
        print(f"quant xlsx 없음: {qpath}", file=sys.stderr)
        return 1

    P, Y, A = load_quant_run_tables(qpath)
    dispute = load_dispute_overrides(qpath)
    runs = load_json_runs(jpath)
    concept_mat = best_concept_matrix(runs)

    consensus: dict[tuple[str, int], float] = {}
    source_ids = None
    if baseline is not None and baseline.is_file():
        consensus = load_consensus_final(baseline)
        source_ids = load_optional_source_ids(baseline)
        print(f"합의(최종) 키 (entity, concept): {len(consensus)} (from {baseline.name})")
    else:
        print(
            "경고: baseline xlsx 없음 — 합의 최종 통일 생략, 평가상이만 적용",
            file=sys.stderr,
        )

    df_scores, nc_app, nc_ch, nd_app, nd_ch = build_matrix(
        P, Y, A, concept_mat, consensus, dispute, source_ids
    )
    df_audit = build_concept_audit(concept_mat)
    df_summary = build_summary_final_sheet(df_scores)
    sheet_p = build_single_evaluator_run_sheet(df_scores, "P")
    sheet_y = build_single_evaluator_run_sheet(df_scores, "Y")
    sheet_a = build_single_evaluator_run_sheet(df_scores, "A")

    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df_summary.to_excel(writer, index=False, sheet_name="최종 종합 평가")
        sheet_p.to_excel(writer, index=False, sheet_name="P_measurement")
        sheet_y.to_excel(writer, index=False, sheet_name="Y_measurement")
        sheet_a.to_excel(writer, index=False, sheet_name="A_measurement")
        df_audit.to_excel(writer, index=False, sheet_name="best_concept_per_run")

    score_cols = [c for c in df_scores.columns if re.match(r"^[PYA]_run\d{2}$", c)]
    filled = df_scores[score_cols].notna().sum().sum()
    total = len(df_scores) * len(score_cols)
    print(f"저장: {out}")
    print(f"점수 칸 채움: {filled}/{total} ({100.0 * filled / total:.1f}%)")
    print(
        f"합의 최종 적용 칸(P+Y+A): {nc_app} (값 변경: {nc_ch})"
    )
    print(
        f"평가상이 조정 적용 칸: {nd_app} (값 변경: {nd_ch}); 평가상이 키 수: {len(dispute)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
