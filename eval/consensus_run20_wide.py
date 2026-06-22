"""
run1~run20 합의 표(consensus_run20).

채우는 순서 (run r, 행 i):
  1) best_concept_id 가 있고 (entity, concept_id) 가 인간 합의 맵에 있으면 그 점수.
  2) 아니면 해당 행의 P/Y/A run 점수가 모두 같고(결측 없음) 그 값을 사용 — 처음부터 셋이 동의한 경우.
  3) 그 외 비움(pd.NA).

다수결·평균은 사용하지 않음.

인간 합의 출처(빌드 스크립트에서 딕셔너리로 합침):
  - 평가자 P만 불일치 시: baseline 「최종」
  - SNOMED: 추가로 3_SNOMED_filtering_회의후.xlsx 「평가상이_필터후」「comment」
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

NUM_RUNS = 20
EVALUATORS = ("P", "Y", "A")


def _parse_trailing_concept_id(concept_cell) -> int | None:
    if concept_cell is None or (isinstance(concept_cell, float) and pd.isna(concept_cell)):
        return None
    m = re.search(r"\((\d+)\)\s*$", str(concept_cell).strip())
    return int(m.group(1)) if m else None


def load_snomed_filtering_final(xlsx: Path, domain: str) -> dict[tuple[str, int], float]:
    """회의 후 필터 시트: domain_id 일치 행의 comment = 합의 점수."""
    df = pd.read_excel(xlsx, sheet_name="평가상이_필터후")
    want = domain.casefold()
    out: dict[tuple[str, int], float] = {}
    for _, row in df.iterrows():
        if str(row["domain_id"]).strip().casefold() != want:
            continue
        ek = str(row["entity_name"]).strip().casefold()
        cid = _parse_trailing_concept_id(row.get("concept"))
        if cid is None:
            continue
        v = row.get("comment")
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        out[(ek, cid)] = float(v)
    return out


def resolve_snomed_filtering_path(root: Path) -> Path | None:
    """NFC/NFD 파일명 둘 중 존재하는 경로."""
    for name in (
        "3_SNOMED_filtering_회의후.xlsx",
        "3_SNOMED_filtering_회의후.xlsx",
    ):
        p = root / "human-test" / name
        if p.is_file():
            return p
    return None


def _unanimous_pya_score(df_scores: pd.DataFrame, row_index: int, run_label: int) -> float | None:
    """P_runXX, Y_runXX, A_runXX 가 모두 같고 결측 없으면 그 실수, 아니면 None."""
    cols = [f"{ev}_run{run_label:02d}" for ev in EVALUATORS]
    try:
        vals = [df_scores.iloc[row_index][c] for c in cols]
    except (IndexError, KeyError):
        return None
    if any(pd.isna(v) for v in vals):
        return None
    try:
        a, b, c = (float(vals[0]), float(vals[1]), float(vals[2]))
    except (TypeError, ValueError):
        return None
    if a == b == c:
        return a
    return None


def build_consensus_wide_human_map(
    *,
    entities: list[str],
    domain: str,
    concept_mat: list[list[int | None]],
    human_scores: dict[tuple[str, int], float],
    source_ids: list | None = None,
    df_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    human_scores: (entity casefold, concept_id) -> 합의 점수 (엑셀 「최종」/「comment」 등).
    df_scores: 합의·평가상이 반영 **후** P/Y/A 매트릭스 — 일치 보조용.
    """
    n = len(entities)
    rows: list[dict] = []
    for i in range(n):
        ek = entities[i].strip().casefold()
        row: dict = {
            "index": i + 1,
            "entity_name": entities[i],
            "domain": domain,
        }
        if source_ids is not None and i < len(source_ids):
            row["source_id"] = source_ids[i]
        for r in range(1, NUM_RUNS + 1):
            cid = concept_mat[r - 1][i]
            if cid is not None:
                v = human_scores.get((ek, cid))
                if v is not None:
                    row[f"run{r}"] = v
                    continue
            u = _unanimous_pya_score(df_scores, i, r) if df_scores is not None else None
            row[f"run{r}"] = u if u is not None else pd.NA
        rows.append(row)

    cols = ["index"]
    if source_ids is not None:
        cols.append("source_id")
    cols.extend(["entity_name", "domain", *[f"run{r}" for r in range(1, NUM_RUNS + 1)]])
    return pd.DataFrame(rows)[cols]
