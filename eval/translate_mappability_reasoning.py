#!/usr/bin/env python3
"""
mappability_baseline 결과 엑셀의 `mappability_reasoning`(영어)을 LLM으로 한국어 번역해
`mappability_reasoning_ko` 컬럼을 추가합니다. 배치(여러 문단 한 번에) + 병렬로 빠르게 처리합니다.

Usage:
    python eval/translate_mappability_reasoning.py --input-dir eval/mappability_baseline_out
    python eval/translate_mappability_reasoning.py -i eval/mappability_baseline_out --batch-size 15 -j 8
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a professional translator for medical informatics and OMOP CDM terminology.
Translate English explanatory text into natural, accurate Korean for clinical data specialists.

Rules:
- Preserve meaning; keep drug names, SNOMED/OMOP concept wording in parentheses when present.
- Do not translate JSON-like tokens or numeric concept IDs; keep them as in the source if they appear inside the prose.
- Output must be ONLY valid JSON with the exact schema requested. No markdown, no commentary."""

USER_BATCH_TEMPLATE = """The input is a JSON object with key "paragraphs": an array of English strings (same order matters).
Translate each paragraph to Korean. Empty strings must remain empty strings in the output.

Input:
{payload}

Respond with JSON only, exactly this shape:
{{"translations": ["한국어...", "..."]}}
The "translations" array MUST have the same length as input "paragraphs"."""


def _thread_llm_client():
    c = getattr(_tls, "client", None)
    if c is None:
        c = get_llm_client()
        _tls.client = c
    return c


def _parse_translations_json(text: str) -> list[str] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return None
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    arr = obj.get("translations")
    if not isinstance(arr, list):
        return None
    return ["" if x is None else str(x) for x in arr]


def translate_batch(paragraphs: list[str]) -> list[str]:
    """paragraphs와 동일 길이의 한국어 문자열 리스트 반환. 실패 시 예외."""
    if not paragraphs:
        return []
    llm = _thread_llm_client()
    payload = json.dumps({"paragraphs": paragraphs}, ensure_ascii=False)
    user = USER_BATCH_TEMPLATE.format(payload=payload)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    resp = llm.chat_completion(
        messages=messages,
        temperature=0.0,
        json_mode=True,
    )
    if not resp:
        raise RuntimeError("empty LLM response")

    out = _parse_translations_json(resp)
    if out is None or len(out) != len(paragraphs):
        raise RuntimeError(
            f"bad translations len: got {len(out) if out else None}, want {len(paragraphs)}"
        )
    return out


def translate_batch_safe(paragraphs: list[str], *, retries: int = 2) -> list[str]:
    """길이 불일치 시 배치를 반으로 쪼개 재시도, 최종적으로 한 줄씩."""
    if not paragraphs:
        return []
    try:
        return translate_batch(paragraphs)
    except Exception as e:
        if len(paragraphs) == 1:
            logger.warning("번역 실패(1문단): %s — 원문 유지", e)
            return [paragraphs[0]]
        if retries <= 0:
            raise
        mid = max(1, len(paragraphs) // 2)
        logger.info("배치 분할 재시도 (%s문단 → %s+%s): %s", len(paragraphs), mid, len(paragraphs) - mid, e)
        return translate_batch_safe(paragraphs[:mid], retries=retries - 1) + translate_batch_safe(
            paragraphs[mid:], retries=retries - 1
        )


def _work_batch(args: tuple[list[int], list[str]]) -> tuple[list[int], list[str]]:
    indices, texts = args
    ko = translate_batch_safe(texts)
    return indices, ko


def process_dataframe(
    df: pd.DataFrame,
    *,
    batch_size: int,
    workers: int,
) -> pd.Series:
    """mappability_reasoning → 한국어 Series (행 순서와 동일)."""
    col = "mappability_reasoning"
    if col not in df.columns:
        raise ValueError(f"컬럼 없음: {col}")

    n = len(df)
    out = pd.Series([""] * n, dtype=object)

    jobs: list[tuple[list[int], list[str]]] = []
    buf_i: list[int] = []
    buf_t: list[str] = []

    for i in range(n):
        raw = df.iloc[i][col]
        s = "" if pd.isna(raw) else str(raw).strip()
        if not s:
            out.iloc[i] = ""
            continue
        buf_i.append(i)
        buf_t.append(s)
        if len(buf_t) >= batch_size:
            jobs.append((buf_i, buf_t))
            buf_i, buf_t = [], []

    if buf_t:
        jobs.append((buf_i, buf_t))

    if not jobs:
        return out

    logger.info("번역 배치 %s개 (workers=%s, batch_size≤%s)", len(jobs), workers, batch_size)

    if workers <= 1:
        for indices, texts in jobs:
            _, kos = _work_batch((indices, texts))
            for idx, k in zip(indices, kos):
                out.iloc[idx] = k
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_work_batch, job) for job in jobs]
            for fut in as_completed(futs):
                indices, kos = fut.result()
                for idx, k in zip(indices, kos):
                    out.iloc[idx] = k

    return out


def run_file(
    path: Path,
    *,
    batch_size: int,
    workers: int,
    dry_run: bool,
) -> None:
    df = pd.read_excel(path)
    if dry_run:
        if "mappability_reasoning" not in df.columns:
            logger.warning("[dry-run] %s — mappability_reasoning 없음", path.name)
            return
        s = df["mappability_reasoning"]
        nonempty = int((s.notna() & (s.astype(str).str.strip() != "")).sum())
        est_batches = (nonempty + batch_size - 1) // batch_size if nonempty else 0
        logger.info("[dry-run] %s — 번역 대상 약 %s행 → 예상 배치 ~%s개", path.name, nonempty, est_batches)
        return

    series_ko = process_dataframe(df, batch_size=batch_size, workers=workers)
    df["mappability_reasoning_ko"] = series_ko

    tmp = path.with_suffix(".tmp.xlsx")
    sheet = (path.stem[:31] or "data").replace("[", "").replace("]", "").replace("*", "")
    with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet)
    tmp.replace(path)
    logger.info("저장 완료: %s (%s행)", path, len(df))


def write_combined_from_sources(out_dir: Path, combined_name: str) -> None:
    parts = sorted(out_dir.glob("evaluation_blind_test_*_mappability_baseline.xlsx"))
    if not parts:
        logger.warning("통합용 소스 파일(evaluation_blind_test_*...) 없음")
        return
    dfs = [pd.read_excel(p) for p in parts]
    merged = pd.concat(dfs, ignore_index=True)
    out_path = out_dir / combined_name
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        merged.to_excel(writer, index=False, sheet_name="mappability_all")
    logger.info("통합 파일 작성: %s (%s행)", out_path, len(merged))


def main() -> None:
    ap = argparse.ArgumentParser(description="mappability_reasoning → 한국어 컬럼 추가")
    ap.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        default=_root / "eval" / "mappability_baseline_out",
        help="엑셀이 있는 폴더",
    )
    ap.add_argument(
        "--pattern",
        default="*_mappability_baseline.xlsx",
        help="처리할 파일 glob (폴더 기준)",
    )
    ap.add_argument(
        "--include-combined",
        action="store_true",
        help="mappability_baseline_combined.xlsx 도 번역 (기본은 제외해 중복 호출 방지)",
    )
    ap.add_argument("--batch-size", type=int, default=12, help="한 번의 API 호출에 넣을 문단 수")
    ap.add_argument("-j", "--workers", type=int, default=8, help="동시 배치 수")
    ap.add_argument("--dry-run", action="store_true", help="API 없이 배치 수만 로그")
    ap.add_argument(
        "--write-combined",
        action="store_true",
        default=True,
        help="소스 3개 처리 후 mappability_baseline_combined.xlsx 재생성 (기본 True)",
    )
    ap.add_argument(
        "--no-write-combined",
        action="store_true",
        help="통합 파일 재생성 안 함",
    )
    ap.add_argument("--combined-name", default="mappability_baseline_combined.xlsx")
    args = ap.parse_args()

    d = args.input_dir.resolve()
    files = sorted(d.glob(args.pattern))
    if not files:
        raise SystemExit(f"파일 없음: {d}/{args.pattern}")

    if not args.include_combined:
        files = [f for f in files if f.name != args.combined_name]

    logger.info("대상 %s개: %s", len(files), ", ".join(f.name for f in files))

    llm = get_llm_client()
    if not args.dry_run and not llm.is_initialized:
        raise RuntimeError("LLM 클라이언트 초기화 실패 (.env OPENAI_API_KEY 등)")

    for f in files:
        logger.info("번역 중: %s", f.name)
        run_file(f, batch_size=args.batch_size, workers=args.workers, dry_run=args.dry_run)

    wc = args.write_combined and not args.no_write_combined
    if wc and not args.dry_run:
        write_combined_from_sources(d, args.combined_name)


if __name__ == "__main__":
    main()
