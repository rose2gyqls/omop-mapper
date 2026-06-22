#!/usr/bin/env python3
"""
범용: comparison_baseline 행 순서 + JSON 20 runs + quant P/Y/A 시트 + 합의/평가상이.

- JSON 행은 row_id(SNUH) 또는 id(SNOMED)로 정렬 후, input_domain 필터(선택)로 자른 목록이
  baseline 의 source_value 행과 1:1 엔티티 문자열로 일치해야 함.
- quant는 엔티티 이름(casefold)으로 행 조회(행 수가 baseline과 다를 수 있음).
- 합의(인간) → (entity, concept) run 일치 시 P=Y=A=합의 점수. SNOMED는 baseline 「최종」∪ 회의후 필터 「comment」(중복 시 최종 우선). 평가상이는 동 키 있으면 스킵.
- 시트 consensus_run20 은 P/Y/A 산술 없이, 위 합의 맵과 run별 best_concept_id 조합으로만 채움(없으면 비움).

서브커맨드: snuh-procedure | snuh-condition | snomed-measurement | …

Usage 예:
  python eval/build_eval_run_matrix.py snuh-condition
  python eval/build_eval_run_matrix.py snomed-measurement
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_eval_dir = Path(__file__).resolve().parent
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

import pandas as pd

from consensus_run20_wide import (
    build_consensus_wide_human_map,
    load_snomed_filtering_final,
    resolve_snomed_filtering_path,
)

NUM_RUNS = 20
EVALUATORS = ("P", "Y", "A")
SHEET_DISPUTE = "평가상이_목록"
SHEET_CONSENSUS = "평가점수_일치작업_평가자P만다름"
SHEET_METHODS = "comparison_baseline"
SNUH_DOMAINS = ("Condition", "Measurement", "Procedure", "Drug")
SNOMED_DOMAINS = ("Condition", "Measurement", "Procedure", "Observation")


@dataclass(frozen=True)
class RunMatrixJob:
    baseline: Path
    json_path: Path
    quant_path: Path
    p_sheet: str
    y_sheet: str
    a_sheet: str
    dispute_quant: Path
    dispute_domain: str
    dispute_dataset_substr: str | None
    json_id_key: Literal["row_id", "id"]
    json_domain_filter: str | None
    metrics_dataset: Literal["SNUH", "SNOMED"]
    fill_domain: str
    total_accuracy_dataset: Literal["SNUH", "SNOMED"]
    metrics_kind: Literal["3pt", "4pt"]
    output: Path
    json_align: Literal["index", "entity_name"] = "index"


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_trailing_concept_id(concept_cell) -> int | None:
    if concept_cell is None or (isinstance(concept_cell, float) and pd.isna(concept_cell)):
        return None
    m = re.search(r"\((\d+)\)\s*$", str(concept_cell).strip())
    return int(m.group(1)) if m else None


def load_consensus_final(baseline_xlsx: Path) -> dict[tuple[str, int], float]:
    df = pd.read_excel(baseline_xlsx, sheet_name=SHEET_CONSENSUS)
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


def load_dispute(
    quant_xlsx: Path,
    *,
    domain: str,
    dataset_substr: str | None,
) -> dict[tuple[str, int], dict[str, float | None]]:
    df = pd.read_excel(quant_xlsx, sheet_name=SHEET_DISPUTE)
    dom = df["domain_id"].astype(str).str.strip().str.casefold()
    df = df.loc[dom == domain.casefold()].copy()
    if dataset_substr and "dataset" in df.columns:
        m = df["dataset"].astype(str).str.contains(dataset_substr, case=False, na=False)
        df = df.loc[m]
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


def load_comparison_entities(baseline: Path) -> tuple[pd.DataFrame, list[str]]:
    cx = pd.read_excel(baseline, sheet_name=SHEET_METHODS, header=1)
    entities = [str(x).strip() for x in cx["source_value"]]
    return cx, entities


def load_optional_source_ids(baseline_xlsx: Path, n_rows: int) -> list | None:
    df = pd.read_excel(baseline_xlsx, sheet_name=SHEET_METHODS, header=1)
    if "source_id" not in df.columns or len(df) != n_rows:
        return None
    return [df.iloc[i]["source_id"] for i in range(n_rows)]


def load_json_runs(json_path: Path) -> list[list[dict]]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    runs = data.get("runs")
    if not runs or len(runs) != NUM_RUNS:
        raise ValueError(f"runs 기대 {NUM_RUNS}, 실제 {len(runs) if runs else 0}")
    return runs


def filter_runs_by_domain(
    runs: list[list[dict]],
    *,
    id_key: str,
    domain: str | None,
) -> list[list[dict]]:
    out: list[list[dict]] = []
    dom_cf = domain.casefold() if domain else None
    for run in runs:
        items = sorted(run, key=lambda x: int(x[id_key]))
        if dom_cf:
            items = [x for x in items if str(x.get("input_domain", "")).casefold() == dom_cf]
        out.append(items)
    return out


def align_entities(
    entities: list[str],
    json_run0: list[dict],
    *,
    mode: Literal["index", "entity_name"] = "index",
) -> None:
    if mode == "index":
        if len(entities) != len(json_run0):
            raise ValueError(f"행 수 불일치: baseline {len(entities)}, json 필터 {len(json_run0)}")
        for i, (e, item) in enumerate(zip(entities, json_run0)):
            a = e.strip().casefold()
            b = str(item["entity_name"]).strip().casefold()
            if a != b:
                raise ValueError(f"행 {i}: xlsx {e!r} vs json {item['entity_name']!r}")
        return

    by_name: dict[str, dict] = {}
    for item in json_run0:
        k = str(item["entity_name"]).strip().casefold()
        if k not in by_name:
            by_name[k] = item
    missing = [e for e in entities if e.strip().casefold() not in by_name]
    if missing:
        raise ValueError(
            f"JSON에 없는 baseline 엔티티 {len(missing)}건 (예: {missing[0]!r})"
        )


def _parse_best_concept_id(raw) -> int | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("none", "nan"):
        return None
    return int(s)


def best_concept_matrix(
    filtered_runs: list[list[dict]],
    n_rows: int,
    *,
    entities: list[str] | None = None,
    align: Literal["index", "entity_name"] = "index",
) -> list[list[int | None]]:
    mat: list[list[int | None]] = []
    for run in filtered_runs:
        rowc: list[int | None] = []
        if align == "entity_name":
            assert entities is not None
            by_name = {str(x["entity_name"]).strip().casefold(): x for x in run}
            for e in entities:
                item = by_name.get(e.strip().casefold())
                rowc.append(
                    _parse_best_concept_id(item.get("best_concept_id")) if item else None
                )
        else:
            for i in range(n_rows):
                rowc.append(_parse_best_concept_id(run[i].get("best_concept_id")))
        mat.append(rowc)
    return mat


def _quant_lookup(df: pd.DataFrame) -> dict[str, pd.Series]:
    d: dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        k = str(row["entity_name"]).strip().casefold()
        if k not in d:
            d[k] = row
    return d


def _cell_changed(old_v, new_v: float) -> bool:
    if old_v is pd.NA or (isinstance(old_v, (int, float)) and pd.isna(old_v)):
        return True
    try:
        return float(old_v) != float(new_v)
    except (TypeError, ValueError):
        return True


def build_matrix(
    entities: list[str],
    lookups: dict[str, dict[str, pd.Series]],
    concept_mat: list[list[int | None]],
    consensus: dict[tuple[str, int], float],
    dispute: dict[tuple[str, int], dict[str, float | None]],
    domain_col: str,
    source_ids: list | None,
) -> tuple[pd.DataFrame, int, int, int, int]:
    n = len(entities)
    rows: list[dict] = []
    n_cons_apply = n_cons_changed = n_dis_apply = n_dis_changed = 0

    for i in range(n):
        entity = entities[i]
        ek = entity.casefold()
        base: dict = {
            "index": i + 1,
            "entity_name": entity,
            "domain_id": domain_col,
        }
        if source_ids is not None:
            base["source_id"] = source_ids[i]

        for ev in EVALUATORS:
            lk = lookups[ev]
            row = lk.get(ek)
            for r in range(NUM_RUNS):
                col = f"{ev}_run{r + 1:02d}"
                if row is None:
                    base[col] = pd.NA
                    continue
                v = row.get(f"run{r + 1}")
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


def build_concept_audit(concept_mat: list[list[int | None]], n_rows: int) -> pd.DataFrame:
    data = {"index": list(range(1, n_rows + 1))}
    for r in range(NUM_RUNS):
        data[f"best_concept_id_run{r + 1:02d}"] = [concept_mat[r][i] for i in range(n_rows)]
    return pd.DataFrame(data)


def _acc2_weighted_3pt(flat_scores: list[float]) -> tuple[float, float]:
    if not flat_scores:
        return float("nan"), float("nan")
    n = len(flat_scores)
    acc_2 = sum(1.0 for s in flat_scores if s >= 2.0) / n * 100.0
    weighted = (
        sum(1.0 if s >= 2.0 else (0.5 if s >= 1.0 else 0.0) for s in flat_scores) / n * 100.0
    )
    return acc_2, weighted


def _acc_drug_4pt(flat_scores: list[float]) -> tuple[float, float, float]:
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


def _eval_metrics_3pt(df_scores: pd.DataFrame) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for ev in EVALUATORS:
        flat: list[float] = []
        for r in range(NUM_RUNS):
            for v in df_scores[f"{ev}_run{r + 1:02d}"]:
                if pd.notna(v):
                    flat.append(float(v))
        out[ev] = _acc2_weighted_3pt(flat)
    return out


def _eval_metrics_4pt(df_scores: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    for ev in EVALUATORS:
        flat = []
        for r in range(NUM_RUNS):
            for v in df_scores[f"{ev}_run{r + 1:02d}"]:
                if pd.notna(v):
                    flat.append(float(v))
        out[ev] = _acc_drug_4pt(flat)
    return out


def build_summary(
    df_scores: pd.DataFrame,
    *,
    metrics_kind: Literal["3pt", "4pt"],
    fill_domain: str,
    metrics_dataset: Literal["SNUH", "SNOMED"],
    total_accuracy_dataset: Literal["SNUH", "SNOMED"],
) -> pd.DataFrame:
    if metrics_kind == "3pt":
        m = _eval_metrics_3pt(df_scores)
        dom_acc = sum(m[ev][0] for ev in EVALUATORS) / 3.0
    else:
        m4 = _eval_metrics_4pt(df_scores)
        dom_acc = sum(m4[ev][0] for ev in EVALUATORS) / 3.0

    rows: list[dict[str, object]] = []

    def _block(dataset_name: str, domains: tuple[str, ...]) -> None:
        nonlocal rows
        for di, domain in enumerate(domains):
            for ei, ev in enumerate(EVALUATORS):
                is_first_in_domain = ei == 0
                is_first_row_ds = di == 0 and ei == 0

                acc2 = wavg = acc21 = dom_cell = total_cell = pd.NA

                if is_first_row_ds and dataset_name == total_accuracy_dataset:
                    total_cell = round(dom_acc, 2)

                if (
                    domain == fill_domain
                    and dataset_name == metrics_dataset
                ):
                    if metrics_kind == "3pt":
                        a2, w = m[ev]
                        acc2, wavg = round(a2, 2), round(w, 2)
                    else:
                        a2, a21, w = m4[ev]
                        acc2, acc21, wavg = round(a2, 2), round(a21, 2), round(w, 2)
                    if is_first_in_domain:
                        dom_cell = round(dom_acc, 2)

                rows.append(
                    {
                        "데이터": dataset_name if is_first_row_ds else pd.NA,
                        "도메인": domain if is_first_in_domain else pd.NA,
                        "평가자": ev,
                        "Acc_2(%)": acc2,
                        "Acc_2+1(%, drug)": acc21,
                        "가중평균(%)": wavg,
                        "도메인 정확도": dom_cell,
                        "총 정확도": total_cell,
                    }
                )

    _block("SNUH", SNUH_DOMAINS)
    _block("SNOMED", SNOMED_DOMAINS)
    assert len(rows) == 24
    return pd.DataFrame(rows)


def build_single_eval_sheet(df_scores: pd.DataFrame, ev: str, sheet_name: str) -> pd.DataFrame:
    base_cols = ["index", "entity_name"]
    if "source_id" in df_scores.columns:
        base_cols.insert(1, "source_id")
    part = df_scores[base_cols + ["domain_id"]].copy().rename(columns={"domain_id": "domain"})
    rename_runs = {f"{ev}_run{r + 1:02d}": f"run{r + 1}" for r in range(NUM_RUNS)}
    runs = df_scores[[f"{ev}_run{r + 1:02d}" for r in range(NUM_RUNS)]].rename(columns=rename_runs)
    out = pd.concat([part.reset_index(drop=True), runs.reset_index(drop=True)], axis=1)
    out.attrs["sheet_name"] = sheet_name
    return out


def run_job(job: RunMatrixJob) -> int:
    if not job.baseline.is_file():
        print(f"baseline 없음: {job.baseline}", file=sys.stderr)
        return 1

    jpath = job.json_path.expanduser().resolve()
    if not jpath.is_file():
        print(f"JSON 없음: {jpath}", file=sys.stderr)
        return 1

    cx, entities = load_comparison_entities(job.baseline.resolve())
    n_rows = len(entities)
    domain_col = str(cx.iloc[0]["domain_id"]) if n_rows else "unknown"

    raw_runs = load_json_runs(jpath)
    filtered = filter_runs_by_domain(
        raw_runs,
        id_key=job.json_id_key,
        domain=job.json_domain_filter,
    )
    align_entities(entities, filtered[0], mode=job.json_align)

    qpath = job.quant_path.expanduser().resolve()
    if not qpath.is_file():
        print(f"quant 없음: {qpath}", file=sys.stderr)
        return 1
    P = pd.read_excel(qpath, sheet_name=job.p_sheet)
    Y = pd.read_excel(qpath, sheet_name=job.y_sheet)
    A = pd.read_excel(qpath, sheet_name=job.a_sheet)
    lookups = {ev: _quant_lookup(df) for ev, df in zip(EVALUATORS, (P, Y, A))}

    concept_mat = best_concept_matrix(
        filtered,
        n_rows,
        entities=entities if job.json_align == "entity_name" else None,
        align=job.json_align,
    )
    consensus = load_consensus_final(job.baseline.resolve())
    if job.metrics_dataset == "SNOMED":
        filt_path = resolve_snomed_filtering_path(_root())
        if filt_path is not None:
            filt = load_snomed_filtering_final(filt_path, domain_col)
            consensus = {**filt, **consensus}
    dq = job.dispute_quant.expanduser().resolve()
    if not dq.is_file():
        print(f"dispute quant 없음: {dq}", file=sys.stderr)
        return 1
    dispute = load_dispute(
        dq,
        domain=job.dispute_domain,
        dataset_substr=job.dispute_dataset_substr,
    )

    source_ids = load_optional_source_ids(job.baseline.resolve(), n_rows)

    df_scores, nc_a, nc_c, nd_a, nd_c = build_matrix(
        entities,
        lookups,
        concept_mat,
        consensus,
        dispute,
        domain_col,
        source_ids,
    )

    summary = build_summary(
        df_scores,
        metrics_kind=job.metrics_kind,
        fill_domain=job.fill_domain,
        metrics_dataset=job.metrics_dataset,
        total_accuracy_dataset=job.total_accuracy_dataset,
    )

    job.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(job.output, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="최종 종합 평가")
        build_consensus_wide_human_map(
            entities=entities,
            domain=domain_col,
            concept_mat=concept_mat,
            human_scores=consensus,
            source_ids=source_ids,
            df_scores=df_scores,
        ).to_excel(writer, index=False, sheet_name="consensus_run20")
        for ev, sn in zip(EVALUATORS, (job.p_sheet, job.y_sheet, job.a_sheet)):
            build_single_eval_sheet(df_scores, ev, sn).to_excel(
                writer, index=False, sheet_name=sn[:31]
            )
        build_concept_audit(concept_mat, n_rows).to_excel(
            writer, index=False, sheet_name="best_concept_per_run"
        )

    score_cols = [c for c in df_scores.columns if re.match(r"^[PYA]_run\d{2}$", c)]
    filled = df_scores[score_cols].notna().sum().sum()
    total = len(df_scores) * len(score_cols)
    print(f"저장: {job.output}")
    print(f"행 {n_rows}, 점수칸 {filled}/{total} ({100*filled/total:.1f}%)")
    print(f"합의 적용 {nc_a} (변경 {nc_c}), 평가상이 {nd_a} (변경 {nd_c})")
    return 0


def main() -> int:
    root = _root()
    ap = argparse.ArgumentParser(description="baseline + JSON + quant run 매트릭스")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_cmd(
        name: str,
        *,
        baseline: Path,
        json_path: Path,
        quant: Path,
        p: str,
        y: str,
        a: str,
        dq: Path,
        dd: str,
        dsub: str | None,
        idk: Literal["row_id", "id"],
        jdom: str | None,
        md: Literal["SNUH", "SNOMED"],
        fd: str,
        tad: Literal["SNUH", "SNOMED"],
        mk: Literal["3pt", "4pt"],
        out: Path,
        json_align: Literal["index", "entity_name"] = "index",
    ) -> None:
        sp = sub.add_parser(name)
        sp.set_defaults(
            _job=RunMatrixJob(
                baseline=baseline,
                json_path=json_path,
                quant_path=quant,
                p_sheet=p,
                y_sheet=y,
                a_sheet=a,
                dispute_quant=dq,
                dispute_domain=dd,
                dispute_dataset_substr=dsub,
                json_id_key=idk,
                json_domain_filter=jdom,
                metrics_dataset=md,
                fill_domain=fd,
                total_accuracy_dataset=tad,
                metrics_kind=mk,
                output=out,
                json_align=json_align,
            )
        )

    ht = root / "human-test"
    add_cmd(
        "snuh-procedure",
        baseline=ht / "SNUH_procedure_평가자P_comparison_baseline.xlsx",
        json_path=root / "test_logs" / "SNUH" / "mapping_snuh_proc_merged.json",
        quant=root / "eval" / "quantitative_evaluation_snuh_meas.xlsx",
        p="P_snuh_procedure",
        y="Y_snuh_procedure",
        a="A_snuh_procedure",
        dq=root / "eval" / "quantitative_evaluation_snuh_meas.xlsx",
        dd="procedure",
        dsub=None,
        idk="row_id",
        jdom=None,
        md="SNUH",
        fd="Procedure",
        tad="SNUH",
        mk="3pt",
        out=root / "eval" / "out" / "SNUH_procedure_237x20_evaluators.xlsx",
    )
    add_cmd(
        "snuh-condition",
        baseline=ht / "SNUH_condition_평가자P_comparison_baseline.xlsx",
        json_path=root / "test_logs" / "SNUH" / "mapping_snuh_20260308_144131_con.json",
        quant=root / "eval" / "old" / "quantitative_evaluation_1.xlsx",
        p="P_snuh_condition",
        y="Y_snuh_condition",
        a="A_snuh_condition",
        dq=root / "eval" / "old" / "quantitative_evaluation_1.xlsx",
        dd="condition",
        dsub="SNUH",
        idk="row_id",
        jdom="Condition",
        md="SNUH",
        fd="Condition",
        tad="SNUH",
        mk="3pt",
        out=root / "eval" / "out" / "SNUH_condition_1578x20_evaluators.xlsx",
        json_align="entity_name",
    )
    add_cmd(
        "snomed-measurement",
        baseline=ht / "SNOMED_measurement_평가자P_comparison_baseline.xlsx",
        json_path=root / "test_logs" / "SNOMED" / "mapping_snomed_merged.json",
        quant=root / "eval" / "quantitative_evaluation.xlsx",
        p="P_snomed_measurement",
        y="Y_snomed_measurement",
        a="A_snomed_measurement",
        dq=root / "eval" / "quantitative_evaluation.xlsx",
        dd="measurement",
        dsub="SNOMED",
        idk="id",
        jdom="Measurement",
        md="SNOMED",
        fd="Measurement",
        tad="SNOMED",
        mk="3pt",
        out=root / "eval" / "out" / "SNOMED_measurement_250x20_evaluators.xlsx",
    )
    add_cmd(
        "snomed-procedure",
        baseline=ht / "SNOMED_procedure_평가자P_comparison_baseline.xlsx",
        json_path=root / "test_logs" / "SNOMED" / "mapping_snomed_merged.json",
        quant=root / "eval" / "quantitative_evaluation.xlsx",
        p="P_snomed_procedure",
        y="Y_snomed_procedure",
        a="A_snomed_procedure",
        dq=root / "eval" / "quantitative_evaluation.xlsx",
        dd="procedure",
        dsub="SNOMED",
        idk="id",
        jdom="Procedure",
        md="SNOMED",
        fd="Procedure",
        tad="SNOMED",
        mk="3pt",
        out=root / "eval" / "out" / "SNOMED_procedure_250x20_evaluators.xlsx",
    )
    add_cmd(
        "snomed-condition",
        baseline=ht / "SNOMED_condition_평가자P_comparison_baseline.xlsx",
        json_path=root / "test_logs" / "SNOMED" / "mapping_snomed_merged.json",
        quant=root / "eval" / "quantitative_evaluation.xlsx",
        p="P_snomed_condition",
        y="Y_snomed_condition",
        a="A_snomed_condition",
        dq=root / "eval" / "quantitative_evaluation.xlsx",
        dd="condition",
        dsub="SNOMED",
        idk="id",
        jdom="Condition",
        md="SNOMED",
        fd="Condition",
        tad="SNOMED",
        mk="3pt",
        out=root / "eval" / "out" / "SNOMED_condition_250x20_evaluators.xlsx",
    )
    add_cmd(
        "snomed-observation",
        baseline=ht / "SNOMED_observation_평가자P_comparison_baseline.xlsx",
        json_path=root / "test_logs" / "SNOMED" / "mapping_snomed_merged.json",
        quant=root / "eval" / "quantitative_evaluation.xlsx",
        p="P_snomed_observation",
        y="Y_snomed_observation",
        a="A_snomed_observation",
        dq=root / "eval" / "quantitative_evaluation.xlsx",
        dd="observation",
        dsub="SNOMED",
        idk="id",
        jdom="Observation",
        md="SNOMED",
        fd="Observation",
        tad="SNOMED",
        mk="3pt",
        out=root / "eval" / "out" / "SNOMED_observation_250x20_evaluators.xlsx",
    )

    args = ap.parse_args()
    job: RunMatrixJob = args._job
    return run_job(job)


if __name__ == "__main__":
    raise SystemExit(main())
