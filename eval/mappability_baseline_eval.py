#!/usr/bin/env python3
"""
최종 평가용 엑셀(entity, domain, candidate concepts)에 대해 GPT로 '매핑 가능성(mappability)' baseline 점수를 부여합니다.

점수 체계:
  0 — 매핑 불가
  1 — 매핑 가능
  5 — 약어
  9 — 정보과다(단일 컨셉으로 매핑 불가)

Usage:
    python eval/mappability_baseline_eval.py --input data/final
    python eval/mappability_baseline_eval.py -i data/final -j 12
    python eval/mappability_baseline_eval.py -i data/final/evaluation_blind_test_1_SNUH_condition_용어.xlsx -o eval/out/mappability
    python eval/mappability_baseline_eval.py -i data/final --limit 5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

import pandas as pd

# MapOMOP 패키지 __init__은 elasticsearch 등 의존을 끌어오므로 llm_client만 직접 로드
_llm_spec = importlib.util.spec_from_file_location(
    "mapomop_llm_client",
    _root / "src" / "MapOMOP" / "llm_client.py",
)
if _llm_spec is None or _llm_spec.loader is None:
    raise RuntimeError("llm_client.py 로드 실패")
_llm_mod = importlib.util.module_from_spec(_llm_spec)
_llm_spec.loader.exec_module(_llm_mod)
get_llm_client = _llm_mod.get_llm_client

_tls = threading.local()


def _thread_llm_client():
    """스레드별 LLM 클라이언트 (병렬 호출 시 공유 상태 충돌 완화)."""
    c = getattr(_tls, "client", None)
    if c is None:
        c = get_llm_client()
        _tls.client = c
    return c


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VALID_SCORES = frozenset({0, 1, 5, 9})

SYSTEM_PROMPT = """You are an expert clinical data scientist specializing in the OMOP Common Data Model (CDM) and medical terminology mapping.
Your task is to evaluate the 'mappability' of a raw medical entity extracted from clinical records.

You will be provided with:
1. Entity: The raw text string that needs to be mapped.
2. Domain: The expected clinical domain of the entity (e.g., Condition, Drug, Measurement).
3. Candidate Concepts: A list of unique OMOP standard concepts that a previous mapping pipeline suggested for this entity.

Based on this information, evaluate whether the 'Entity' is appropriate for mapping to a single standard concept. Assign a score based on the following criteria:

[Scoring Criteria]
- Score 1 (매핑 가능, Mappable): The entity is a clear, distinct, and specific clinical concept that can be unambiguously mapped to a single standard terminology. The candidate concepts generally reflect this clear meaning.
- Score 5 (약어, Abbreviation): The entity is an acronym or abbreviation (e.g., "HTN", "DM", "cbc"). It requires expansion or context to be definitively mapped, even if the candidate concepts guessed it correctly.
- Score 9 (정보과다, Information Overload): The entity contains multiple distinct clinical concepts (post-coordinated), describes a complex procedure, or is a descriptive sentence/phrase. It is impossible to map this entity to a *single* standard concept without losing information.
- Score 0 (매핑 불가, Unmappable): The entity is too vague, contains severe typos making it unrecognizable, is a non-medical term, or represents a concept completely outside the scope of standard clinical vocabularies.

[Output Format]
You must respond ONLY with a valid JSON object. Do not include any conversational filler, markdown formatting (like ```json), or explanations outside the JSON.
Use the exact structure below:
{
  "reasoning": "Step-by-step explanation of why the entity falls into a specific category. Analyze the entity's structure, clarity, and how it relates to the candidate concepts.",
  "score": <integer score: 0, 1, 5, or 9>
}"""


USER_PROMPT_TEMPLATE = """Please evaluate the mappability of the following entity.

Entity: {entity_name}
Domain: {domain_name}
Candidate Concepts:
{concepts_block}

Output JSON:"""


def parse_concept_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if re.match(r"^concept\d+$", str(c))]
    return sorted(cols, key=lambda x: int(re.search(r"\d+", str(x)).group()))


def extract_concepts_from_row(row: pd.Series, concept_cols: list[str]) -> list[str]:
    concepts: list[str] = []
    for col in concept_cols:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            concepts.append(str(val).strip())
    return concepts


def parse_mappability_response(text: str) -> tuple[int | None, str]:
    """LLM 응답에서 score(0,1,5,9)와 reasoning 파싱."""
    raw = (text or "").strip()
    if not raw:
        return None, ""

    obj: dict | None = None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                obj = json.loads(m.group())
            except json.JSONDecodeError:
                pass

    if not isinstance(obj, dict):
        return None, raw[:2000]

    reasoning = obj.get("reasoning")
    score_raw = obj.get("score")
    reasoning_str = str(reasoning).strip() if reasoning is not None else ""

    try:
        s = int(score_raw)
    except (TypeError, ValueError):
        return None, reasoning_str or raw[:2000]

    if s not in VALID_SCORES:
        logger.warning("허용되지 않은 score=%s (0,1,5,9 만 허용)", s)
        return None, reasoning_str or raw[:2000]

    return s, reasoning_str


def evaluate_mappability(
    llm_client,
    entity_name: str,
    domain_name: str,
    concepts: list[str],
) -> tuple[int | None, str, str | None]:
    """
    한 entity에 대해 mappability 평가.
    Returns: (score or None on failure, reasoning, raw_response_or_error)
    """
    if not str(entity_name).strip():
        return None, "", "empty entity_name"

    concepts_block = "\n".join(concepts) if concepts else "(none provided)"

    user_prompt = USER_PROMPT_TEMPLATE.format(
        entity_name=str(entity_name).strip(),
        domain_name=str(domain_name).strip() if pd.notna(domain_name) else "",
        concepts_block=concepts_block,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    response = llm_client.chat_completion(
        messages=messages,
        temperature=0.0,
        json_mode=True,
    )
    if not response:
        return None, "", "LLM returned empty response"

    score, reasoning = parse_mappability_response(response)
    if score is None:
        return None, reasoning, response[:4000]
    return score, reasoning, response


def _build_result_row(
    row: dict,
    concept_cols: list[str],
    llm_client,
) -> dict:
    """한 행 평가 → 결과 dict (원본 컬럼 + mappability_*)."""
    entity_name = row.get("entity_name", "")
    domain_id = row.get("domain_id", "")
    concepts = extract_concepts_from_row(pd.Series(row), concept_cols)

    score, reasoning, raw_or_err = evaluate_mappability(
        llm_client,
        str(entity_name).strip(),
        str(domain_id).strip() if pd.notna(domain_id) else "",
        concepts,
    )

    new_row = dict(row)
    new_row["mappability_score"] = score if score is not None else "ERR"
    new_row["mappability_reasoning"] = reasoning
    if score is None:
        new_row["mappability_raw_or_error"] = raw_or_err
    else:
        new_row["mappability_raw_or_error"] = ""
    return new_row


def _eval_row_task(
    payload: tuple[int, dict, list[str]],
) -> tuple[int, dict]:
    row_idx, row, concept_cols = payload
    return row_idx, _build_result_row(row, concept_cols, _thread_llm_client())


def collect_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".xlsx":
            raise ValueError(f"지원하는 입력은 .xlsx 입니다: {input_path}")
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob("*.xlsx"))
        if not files:
            raise ValueError(f"폴더에 .xlsx 파일이 없습니다: {input_path}")
        return files
    raise FileNotFoundError(str(input_path))


def run_file(
    input_xlsx: Path,
    output_xlsx: Path,
    llm_client,
    *,
    limit: int | None,
    sleep_sec: float,
    workers: int,
    source_file: str | None = None,
) -> pd.DataFrame:
    df = pd.read_excel(input_xlsx)
    concept_cols = parse_concept_columns(df)
    if not concept_cols:
        raise ValueError(f"concept1, concept2, ... 컬럼을 찾을 수 없습니다: {input_xlsx}")

    rows = df.to_dict("records")
    if limit is not None:
        rows = rows[:limit]

    n = len(rows)
    if workers <= 1:
        out_rows: list[dict] = []
        for i, row in enumerate(rows):
            out_rows.append(_build_result_row(row, concept_cols, llm_client))
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            if (i + 1) % 20 == 0:
                logger.info("  %s: %s/%s", input_xlsx.name, i + 1, n)
    else:
        if sleep_sec > 0:
            logger.warning(
                "%s: --sleep 은 --workers>1 일 때 적용되지 않습니다 (병렬).",
                input_xlsx.name,
            )
        logger.info("  %s: 병렬 workers=%s, %s행", input_xlsx.name, workers, n)
        payloads = [(i, rows[i], concept_cols) for i in range(n)]
        out_rows = [None] * n
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_eval_row_task, p): p[0] for p in payloads}
            for fut in as_completed(futures):
                idx, new_row = fut.result()
                out_rows[idx] = new_row
                done += 1
                if done % 20 == 0 or done == n:
                    logger.info("  %s: %s/%s", input_xlsx.name, done, n)

    out_df = pd.DataFrame(out_rows)
    if source_file:
        out_df.insert(0, "source_file", source_file)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name="mappability")
    logger.info("저장: %s (%s행)", output_xlsx, len(out_df))
    return out_df


def summarize_scores(dfs: list[pd.DataFrame]) -> None:
    all_scores: list[int] = []
    err = 0
    for d in dfs:
        for v in d["mappability_score"]:
            if v == "ERR":
                err += 1
            elif isinstance(v, int):
                all_scores.append(v)
    if not all_scores and err == 0:
        return
    logger.info("=" * 50)
    logger.info("mappability 점수 요약 (성공 %s건, ERR %s건)", len(all_scores), err)
    for s in sorted(VALID_SCORES):
        c = sum(1 for x in all_scores if x == s)
        logger.info("  score %s: %s", s, c)
    logger.info("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPT baseline: 평가 엑셀의 entity mappability(0/1/5/9) 판정"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=_root / "data" / "final",
        help="입력 .xlsx 파일 또는 해당 파일들이 있는 폴더 (기본: data/final)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=_root / "eval" / "mappability_baseline_out",
        help="결과 엑셀을 쓸 디렉터리 (기본: eval/mappability_baseline_out)",
    )
    parser.add_argument(
        "--combined-name",
        default="mappability_baseline_combined.xlsx",
        help="통합 결과 파일명 (--no-combined 가 아니면 output-dir 에 저장)",
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="파일별 결과만 저장하고 통합 엑셀은 만들지 않음",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="파일당 평가 행 수 제한 (테스트용)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="순차 모드(workers=1)에서만 API 호출 사이 대기(초)",
    )
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=8,
        metavar="N",
        help="동시 API 요청 수(행 단위 병렬). 1이면 순차 처리 (기본: 8)",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    files = collect_input_files(input_path)

    llm_client = get_llm_client()
    if not llm_client.is_initialized:
        raise RuntimeError(
            "LLM 클라이언트 초기화 실패. OPENAI_API_KEY 및 .env(LLM_*) 설정을 확인하세요."
        )

    out_dir = args.output_dir.resolve()
    combined: list[pd.DataFrame] = []

    logger.info("입력 파일 %s개: %s", len(files), ", ".join(f.name for f in files))
    logger.info("workers=%s (동시 요청 수)", args.workers)

    for fp in files:
        stem = fp.stem
        out_xlsx = out_dir / f"{stem}_mappability_baseline.xlsx"
        logger.info("처리 중: %s", fp.name)
        df_out = run_file(
            fp,
            out_xlsx,
            llm_client,
            limit=args.limit,
            sleep_sec=args.sleep,
            workers=max(1, args.workers),
            source_file=fp.name,
        )
        combined.append(df_out)

    if not args.no_combined and combined:
        merged = pd.concat(combined, ignore_index=True)
        combined_path = out_dir / args.combined_name
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(combined_path, engine="openpyxl") as writer:
            merged.to_excel(writer, index=False, sheet_name="mappability_all")
        logger.info("통합 저장: %s (%s행)", combined_path, len(merged))
        summarize_scores([merged])
    else:
        summarize_scores(combined)


if __name__ == "__main__":
    main()
