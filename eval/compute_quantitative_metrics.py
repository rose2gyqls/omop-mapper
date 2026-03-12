#!/usr/bin/env python3
"""
MapOMOP 정량평가 테이블 생성

20회 매핑 결과 각각을 LLM 평가 점수에 따라 변환:

[기본 3단계: SNOMED 등]
- 2점 → 1 (100%), 1점 → 0.5 (50%), 0점 → 0 (0%)
- Acc_2만(%): 2점만 정답
- 가중평균(%): (1, 0.5, 0) 평균 * 100

[Drug 4단계: SNUH]
- 2, 1, 0.5, 0 점수체계
- Acc_2만(%): 2점만 정답 (2 vs 1+0.5+0)
- Acc_2+1(%): 2점 또는 1점 정답 (2+1 vs 0.5+0)
- 가중평균(%): 2→100%, 1→50%, 0.5→25%, 0→0%

Usage:
    python eval/compute_quantitative_metrics.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import pandas as pd

try:
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


def extract_concept_id(concept_str: str) -> Optional[str]:
    """concept_name(12345) -> 12345"""
    if not concept_str or pd.isna(concept_str):
        return None
    m = re.search(r"\((\d+)\)\s*$", str(concept_str).strip())
    return m.group(1) if m else None


# Drug 도메인: 4단계 점수 (2, 1, 0.5, 0)
DRUG_DOMAIN = "Drug"


def _parse_score(score_raw) -> Optional[float]:
    """LLM 점수 파싱. 2, 1, 0.5, 0 지원."""
    if pd.isna(score_raw):
        return None
    try:
        return float(score_raw)
    except (ValueError, TypeError):
        return None


def build_entity_concept_scores(llm_df: pd.DataFrame) -> dict[int, dict[str, float]]:
    """
    (test_index) -> {concept_id: raw_score}
    raw_score: 2, 1, 0.5, 0 (Drug) 또는 2, 1, 0 (기본)
    """
    concept_cols = [c for c in llm_df.columns if re.match(r"^concept\d+$", c)]
    score_cols = [c for c in llm_df.columns if re.match(r"^llm_score\d+$", c)]

    entity_scores: dict[int, dict[str, float]] = {}
    for _, row in llm_df.iterrows():
        idx = int(row.get("index", row.name))
        entity_scores[idx] = {}
        for cc, sc in zip(concept_cols, score_cols):
            concept = row.get(cc)
            score_raw = row.get(sc)
            if pd.isna(concept) or pd.isna(score_raw) or str(concept).strip() == "":
                continue
            cid = extract_concept_id(str(concept))
            if not cid:
                continue
            s = _parse_score(score_raw)
            if s is not None and s in (0, 0.5, 1, 2):
                entity_scores[idx][cid] = s
    return entity_scores


def load_run_concepts(json_path: str) -> tuple[list[list[dict]], int]:
    """JSON에서 runs 로드. Returns (all_results, num_runs)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "runs" in data:
        return data["runs"], len(data["runs"])
    if isinstance(data, list):
        return [data], 1
    raise ValueError("Unsupported JSON format")


def compute_metrics(
    json_path: str,
    llm_eval_path: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    각 20개 매핑 결과를 2→1, 1→0.5, 0→0 변환 후 풀링하여 도메인별 정확도 계산.
    """
    llm_df = pd.read_excel(llm_eval_path)
    entity_scores = build_entity_concept_scores(llm_df)

    all_results, num_runs = load_run_concepts(json_path)

    # by test_index: [r1, r2, ..., rN]
    by_index: dict[int, list[dict]] = {}
    for run_results in all_results:
        for r in run_results:
            idx = r.get("test_index")
            if idx is None:
                continue
            if idx not in by_index:
                by_index[idx] = []
            by_index[idx].append(r)

    # 풀링: 도메인별 raw_score (2, 1, 0.5, 0) 저장
    # domain -> {"raw_scores": [2, 1, 0.5, ...], "entity_ids": set}
    domain_data: dict[str, dict] = {}
    raw_scores: list[tuple[int, str, list[float]]] = []  # (test_index, domain, [run1_score, ...])

    for test_index, rows in by_index.items():
        for run_idx in range(min(num_runs, len(rows))):
            r = rows[run_idx]
            domain = str(r.get("input_domain", "All")).strip()
            if domain not in domain_data:
                domain_data[domain] = {"raw_scores": [], "entity_ids": set()}

            scores_map = entity_scores.get(test_index, {})
            cid = str(r.get("best_concept_id", "")).strip()
            if not cid and isinstance(r.get("best_concept_id"), (int, float)):
                cid = str(int(r["best_concept_id"]))

            raw_score = scores_map.get(cid)  # 2, 1, 0.5, or 0
            if raw_score is not None:
                domain_data[domain]["raw_scores"].append(raw_score)
            domain_data[domain]["entity_ids"].add(test_index)

        # raw_scores: 각 엔티티별 20 run 점수 (검증용)
        r0 = rows[0]
        domain = str(r0.get("input_domain", "All")).strip()
        scores_map = entity_scores.get(test_index, {})
        entity_scores_list = []
        for run_idx in range(min(num_runs, len(rows))):
            r = rows[run_idx]
            cid = str(r.get("best_concept_id", "")).strip()
            if not cid and isinstance(r.get("best_concept_id"), (int, float)):
                cid = str(int(r["best_concept_id"]))
            s = scores_map.get(cid)
            entity_scores_list.append(s if s is not None else float("nan"))
        raw_scores.append((test_index, domain, entity_scores_list))

    def _compute_domain_metrics(raw_scores_list: list[float], domain: str, use_4pt: bool = False) -> dict:
        """
        도메인별 메트릭 계산.
        Drug 또는 use_4pt: 4단계 (2, 1, 0.5, 0) - Acc_2만, Acc_2+1, 가중평균
        그 외: 3단계 (2, 1, 0) - Acc_2만, 가중평균
        """
        n = len(raw_scores_list)
        if n == 0:
            return {"acc_2": 0, "acc_2_1": None, "weighted": 0}

        is_4pt = domain == DRUG_DOMAIN or use_4pt
        acc_2 = sum(1 for s in raw_scores_list if s >= 2.0) / n * 100

        if is_4pt:
            acc_2_1 = sum(1 for s in raw_scores_list if s >= 1.0) / n * 100
            weighted_sum = sum(
                100 if s >= 2 else (50 if s >= 1 else (25 if s >= 0.5 else 0))
                for s in raw_scores_list
            )
            return {"acc_2": acc_2, "acc_2_1": acc_2_1, "weighted": weighted_sum / n}
        else:
            norm = [1.0 if s >= 2 else (0.5 if s >= 1 else 0.0) for s in raw_scores_list]
            weighted = sum(norm) / n * 100
            return {"acc_2": acc_2, "acc_2_1": None, "weighted": weighted}

    # 테이블: 도메인별 엔티티 수, 총 점수 수, Acc_2만(%), Acc_2+1(%)(Drug만), 가중평균(%)
    domains = sorted(domain_data.keys())
    table_rows = []

    total_entities = len(by_index)
    total_raw = []
    for d in domain_data.values():
        total_raw.extend(d["raw_scores"])

    # 전체: Drug 포함 시 4단계 통일 (0.5 처리)
    has_4pt = DRUG_DOMAIN in domain_data
    total_metrics = _compute_domain_metrics(total_raw, "All", use_4pt=has_4pt)
    has_drug = DRUG_DOMAIN in domain_data

    row = {
        "Scope": "전체",
        "엔티티 수": total_entities,
        "총 점수 수": len(total_raw),
        "Acc_2만(%)": round(total_metrics["acc_2"], 2),
        "가중평균(%)": round(total_metrics["weighted"], 2),
    }
    if has_drug and total_metrics["acc_2_1"] is not None:
        row["Acc_2+1(%)"] = round(total_metrics["acc_2_1"], 2)
    table_rows.append(row)

    for domain in domains:
        d = domain_data[domain]
        raw_list = d["raw_scores"]
        m = _compute_domain_metrics(raw_list, domain)
        row = {
            "Scope": domain,
            "엔티티 수": len(d["entity_ids"]),
            "총 점수 수": len(raw_list),
            "Acc_2만(%)": round(m["acc_2"], 2),
            "가중평균(%)": round(m["weighted"], 2),
        }
        if domain == DRUG_DOMAIN and m["acc_2_1"] is not None:
            row["Acc_2+1(%)"] = round(m["acc_2_1"], 2)
        table_rows.append(row)

    # Acc_2+1 컬럼 통일 (Drug 없으면 빈 열)
    if has_drug:
        for r in table_rows:
            if "Acc_2+1(%)" not in r:
                r["Acc_2+1(%)"] = ""
    rows = table_rows

    df = pd.DataFrame(rows)

    if output_path and HAS_OPENPYXL:
        _save_excel(df, raw_scores, num_runs, output_path)

    return df


def _save_excel(df: pd.DataFrame, raw_scores: list, num_runs: int, output_path: str) -> None:
    """엑셀 저장: 시트1 정량평가, 시트2 점수 매트릭스(검증용)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = openpyxl.Workbook()
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")

    # 시트1: 정량평가 (도메인별 엔티티 수, 정확도)
    ws1 = wb.active
    ws1.title = "정량평가"
    for col, h in enumerate(df.columns, 1):
        c = ws1.cell(row=1, column=col, value=h)
        c.font = header_font
        c.fill = header_fill
    for r_idx, row in enumerate(df.itertuples(index=False), 2):
        for c_idx, val in enumerate(row, 1):
            cell_val = val
            if isinstance(val, float) and val != val:  # nan
                cell_val = ""
            ws1.cell(row=r_idx, column=c_idx, value=cell_val)

    for col in range(1, len(df.columns) + 1):
        ws1.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 14

    # 시트2: 점수 매트릭스 (엔티티 x 20 run) - 1, 0.5, 0 변환 검증용
    ws2 = wb.create_sheet(title="점수매트릭스")
    ws2.cell(row=1, column=1, value="test_index").font = header_font
    ws2.cell(row=1, column=1).fill = header_fill
    ws2.cell(row=1, column=2, value="domain").font = header_font
    ws2.cell(row=1, column=2).fill = header_fill
    for j in range(num_runs):
        ws2.cell(row=1, column=3 + j, value=f"run{j+1}").font = header_font
        ws2.cell(row=1, column=3 + j).fill = header_fill

    for r_idx, (tid, domain, scores) in enumerate(sorted(raw_scores, key=lambda x: x[0]), 2):
        ws2.cell(row=r_idx, column=1, value=tid)
        ws2.cell(row=r_idx, column=2, value=domain)
        for j, s in enumerate(scores):
            val = s if s is not None and s == s else ""  # nan -> ""
            ws2.cell(row=r_idx, column=3 + j, value=val)

    wb.save(output_path)
    print(f"저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MapOMOP 정량평가 테이블 생성")
    parser.add_argument(
        "--json", "-j",
        default=str(_root / "test_logs" / "mapping_snomed_20260311_150447.json"),
        help="매핑 결과 JSON",
    )
    parser.add_argument(
        "--llm-eval", "-l",
        default=str(_root / "eval" / "llm_evaluation_result.xlsx"),
        help="LLM 평가 결과 엑셀",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(_root / "eval" / "quantitative_evaluation.xlsx"),
        help="출력 엑셀 경로",
    )
    args = parser.parse_args()

    df = compute_metrics(
        json_path=args.json,
        llm_eval_path=args.llm_eval,
        output_path=args.output,
    )
    print(df.to_string())


if __name__ == "__main__":
    main()
