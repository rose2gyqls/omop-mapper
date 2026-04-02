"""
CSV에서 domain_id + 텍스트(entity_name 또는 source_value)만 읽어 OMOPHub 검색 결과를 저장합니다.
이전 OMOPHub 출력 CSV에서 오류(429 등) 행만 골라 재시도·병합할 수 있습니다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from tqdm import tqdm

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from omophub.client import OmopHubClient, OmopHubError, SearchHit, _domain_from_row, sleep_ratelimit


def _payload_domain(h: SearchHit) -> str | None:
    """API 응답 행·SearchHit에서 결과 concept의 domain (입력 domain 사용 안 함)."""
    d = h.domain_id
    if d is not None and str(d).strip():
        return str(d).strip()
    if h.raw:
        return _domain_from_row(h.raw)
    return None


def _detect_text_column(df: pd.DataFrame) -> str:
    if "entity_name" in df.columns:
        return "entity_name"
    if "source_value" in df.columns:
        return "source_value"
    raise ValueError(
        "CSV에 entity_name 또는 source_value 열이 필요합니다."
    )


def _row_id_column(df: pd.DataFrame) -> str | None:
    for c in ("test_index", "no", "row_id", "note_id", "snuh_id"):
        if c in df.columns:
            return c
    return None


def _call_omophub(
    text: str,
    domain_str: str,
    client: OmopHubClient,
    *,
    mode: Literal["semantic", "basic"],
    top_k: int,
    vocabulary_ids: str | None,
    threshold: float | None,
    standard_policy: Literal["s_or_c", "s_only", "none"],
    vocab_release: str | None,
) -> tuple[list[SearchHit], str | None]:
    """(hits, error_message). 성공 시 error_message 는 None."""
    if not text.strip():
        return [], "empty_text"
    try:
        hits = client.search(
            text,
            mode=mode,
            domain_id=domain_str or None,
            vocabulary_ids=vocabulary_ids,
            top_k=top_k,
            threshold=threshold,
            standard_policy=standard_policy,
            vocab_release=vocab_release,
        )
        return hits, None
    except OmopHubError as e:
        return [], str(e)


def _hits_to_columns(
    hits: list[SearchHit],
    standard_policy: Literal["s_or_c", "s_only", "none"],
    error: str | None,
) -> dict[str, Any]:
    payload = []
    for h in hits:
        pdomain = _payload_domain(h)
        payload.append(
            {
                "concept_id": h.concept_id,
                "concept_name": h.concept_name,
                "score": h.score,
                "vocabulary_id": h.vocabulary_id,
                "domain_id": pdomain,
                "concept_code": h.concept_code,
                "standard_concept": h.standard_concept,
            }
        )
    out: dict[str, Any] = {
        "top_hits_json": json.dumps(payload, ensure_ascii=False),
        "standard_policy": standard_policy,
    }
    if error is not None:
        out["error"] = error
        out["result1_concept_id"] = None
        out["result1_concept_name"] = None
        out["result1_domain_id"] = None
        out["result1_score"] = None
        out["result1_vocabulary_id"] = None
        out["result1_standard_concept"] = None
        return out

    out["error"] = None
    if hits:
        top = hits[0]
        out["result1_concept_id"] = top.concept_id
        out["result1_concept_name"] = top.concept_name
        out["result1_domain_id"] = payload[0].get("domain_id") if payload else None
        out["result1_score"] = top.score
        out["result1_vocabulary_id"] = top.vocabulary_id
        out["result1_standard_concept"] = top.standard_concept
    else:
        out["result1_concept_id"] = None
        out["result1_concept_name"] = None
        out["result1_domain_id"] = None
        out["result1_score"] = None
        out["result1_vocabulary_id"] = None
        out["result1_standard_concept"] = None
    return out


def run_csv(
    csv_path: Path,
    out_path: Path,
    *,
    mode: Literal["semantic", "basic"] = "semantic",
    top_k: int = 5,
    vocabulary_ids: str | None = None,
    threshold: float | None = None,
    standard_policy: Literal["s_or_c", "s_only", "none"] = "s_or_c",
    sleep_sec: float = 0.0,
    vocab_release: str | None = None,
) -> Path:
    df = pd.read_csv(csv_path)
    if "domain_id" not in df.columns:
        raise ValueError("CSV에 domain_id 열이 필요합니다.")

    text_col = _detect_text_column(df)
    id_col = _row_id_column(df)

    client = OmopHubClient()

    rows_out: list[dict[str, Any]] = []
    for idx in tqdm(range(len(df)), desc=f"OMOPHub {csv_path.name}"):
        row = df.iloc[idx]
        domain_raw = row.get("domain_id")
        domain_str = (
            str(domain_raw).strip()
            if pd.notna(domain_raw) and str(domain_raw).strip() != "nan"
            else ""
        )
        text_raw = row.get(text_col)
        text = str(text_raw).strip() if pd.notna(text_raw) else ""

        rec: dict[str, Any] = {
            "row_index": idx,
            "domain_id": domain_str,
            "source_value": text,
        }
        if id_col is not None:
            v = row.get(id_col)
            rec["source_id"] = v if pd.notna(v) else None

        for gt in ("concept_id", "concept_name", "baseline", "omop_concept_id"):
            if gt in df.columns:
                rec[f"ground_truth_{gt}"] = row.get(gt)

        if not text:
            rec.update(_hits_to_columns([], standard_policy, "empty_text"))
            rows_out.append(rec)
            continue

        hits, err = _call_omophub(
            text,
            domain_str,
            client,
            mode=mode,
            top_k=top_k,
            vocabulary_ids=vocabulary_ids,
            threshold=threshold,
            standard_policy=standard_policy,
            vocab_release=vocab_release,
        )
        rec.update(_hits_to_columns(hits, standard_policy, err))
        rows_out.append(rec)
        sleep_ratelimit(sleep_sec)

    out = pd.DataFrame(rows_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


_RE_RATE_LIMIT = re.compile(
    r"429|rate_limit|rate-limit|too many requests|success\s*:\s*false",
    re.IGNORECASE,
)


def _row_needs_retry(
    err_text: str,
    match: Literal["rate_limit", "any_error"],
) -> bool:
    s = (err_text or "").strip()
    if not s:
        return False
    if match == "any_error":
        return True
    return bool(_RE_RATE_LIMIT.search(s))


def retry_failed_from_output(
    output_csv: Path,
    out_path: Path,
    *,
    match: Literal["rate_limit", "any_error"] = "rate_limit",
    mode: Literal["semantic", "basic"] = "semantic",
    top_k: int = 5,
    vocabulary_ids: str | None = None,
    threshold: float | None = None,
    standard_policy: Literal["s_or_c", "s_only", "none"] | None = None,
    sleep_sec: float = 1.05,
    vocab_release: str | None = None,
) -> tuple[Path, int, int]:
    """
    기존 OMOPHub 출력 CSV를 읽어, error 열이 있는 행만 재요청하고 동일 행 순서로 병합해 저장합니다.

    Returns:
        (저장 경로, 재시도 건수, 전체 행 수)
    """
    df = pd.read_csv(output_csv)
    n_total = len(df)
    if "domain_id" not in df.columns or "source_value" not in df.columns:
        raise ValueError(
            "OMOPHub 출력 CSV에는 domain_id, source_value 열이 있어야 합니다."
        )

    err_col = "error" if "error" in df.columns else None
    err_series = (
        df[err_col].fillna("").astype(str)
        if err_col
        else pd.Series([""] * n_total)
    )

    if standard_policy is not None:
        pol: Literal["s_or_c", "s_only", "none"] = standard_policy
    elif "standard_policy" in df.columns and n_total > 0:
        raw = str(df["standard_policy"].iloc[0]).strip()
        pol = raw if raw in ("s_or_c", "s_only", "none") else "s_or_c"
    else:
        pol = "s_or_c"

    to_retry_mask = pd.Series(
        [_row_needs_retry(err_series.iloc[i], match) for i in range(n_total)],
        dtype=bool,
    )
    n_retry = int(to_retry_mask.sum())

    client = OmopHubClient()

    for i in tqdm(
        [j for j in range(n_total) if to_retry_mask.iloc[j]],
        desc=f"재시도 ({output_csv.name})",
    ):
        row = df.iloc[i]
        domain_raw = row.get("domain_id")
        domain_str = (
            str(domain_raw).strip()
            if pd.notna(domain_raw) and str(domain_raw).strip() != "nan"
            else ""
        )
        text_raw = row.get("source_value")
        text = str(text_raw).strip() if pd.notna(text_raw) else ""

        if not text:
            cols = _hits_to_columns([], pol, "empty_text")
            for k, v in cols.items():
                df.at[i, k] = v
            sleep_ratelimit(sleep_sec)
            continue

        hits, err = _call_omophub(
            text,
            domain_str,
            client,
            mode=mode,
            top_k=top_k,
            vocabulary_ids=vocabulary_ids,
            threshold=threshold,
            standard_policy=pol,
            vocab_release=vocab_release,
        )
        cols = _hits_to_columns(hits, pol, err)
        for k, v in cols.items():
            df.at[i, k] = v
        sleep_ratelimit(sleep_sec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path, n_retry, n_total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OMOPHub로 CSV(domain_id + entity_name|source_value) 배치 검색, "
        "또는 이전 출력 CSV에서 실패 행만 재시도"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, default=None, help="원본 데이터 CSV")
    src.add_argument(
        "--retry-from-output",
        type=Path,
        default=None,
        dest="retry_from_output",
        help="이전 OMOPHub 출력 CSV (error 가 있는 행만 API 재호출 후 병합)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="출력 CSV (미지정 시 outputs/omophub/ 아래 타임스탬프 파일)",
    )
    parser.add_argument(
        "--mode",
        choices=("semantic", "basic"),
        default="semantic",
        help="semantic: 자연어 의미 검색(기본), basic: 키워드 검색",
    )
    parser.add_argument("--top-k", type=int, default=5, dest="top_k")
    parser.add_argument(
        "--vocabulary-ids",
        type=str,
        default=None,
        help="예: SNOMED 또는 SNOMED,LOINC (쉼표 구분)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="semantic 전용, 유사도 하한 (예: 0.3~0.7)",
    )
    parser.add_argument(
        "--standard-concept-policy",
        choices=("s_or_c", "s_only", "none"),
        default=None,
        dest="standard_policy",
        help="미지정 시 재시도 모드에서는 출력 CSV의 standard_policy 열 값을 사용",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=None,
        help="요청 간 초 단위 대기. 재시도 모드에서 미지정이면 기본 1.05 (분당 약 60회 이하 완화)",
    )
    parser.add_argument("--vocab-release", type=str, default=None, dest="vocab_release")
    parser.add_argument(
        "--retry-match",
        choices=("rate_limit", "any_error"),
        default="rate_limit",
        dest="retry_match",
        help="재시도 대상: rate_limit=429·rate_limit·success:false 등(기본), any_error=error 열이 비어 있지 않은 모든 행",
    )
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(_root / ".env")
    except ImportError:
        pass

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.retry_from_output:
        sleep_sec = args.sleep if args.sleep is not None else 1.05
        stem = args.retry_from_output.stem
        out = args.out or (
            _root
            / "outputs"
            / "omophub"
            / f"{stem}_retry_{args.retry_match}_{ts}.csv"
        )
        path, n_retry, n_total = retry_failed_from_output(
            args.retry_from_output.resolve(),
            out.resolve(),
            match=args.retry_match,
            mode=args.mode,
            top_k=args.top_k,
            vocabulary_ids=args.vocabulary_ids,
            threshold=args.threshold,
            standard_policy=args.standard_policy,
            sleep_sec=sleep_sec,
            vocab_release=args.vocab_release,
        )
        print(f"저장 완료: {path}")
        print(f"재시도 {n_retry}건 / 전체 {n_total}건 (매칭: {args.retry_match})")
        return

    assert args.csv is not None
    sleep_sec = args.sleep if args.sleep is not None else 0.0
    pol = args.standard_policy if args.standard_policy is not None else "s_or_c"
    stem = args.csv.stem
    out = args.out or (_root / "outputs" / "omophub" / f"{stem}_omophub_{args.mode}_{ts}.csv")

    path = run_csv(
        args.csv.resolve(),
        out.resolve(),
        mode=args.mode,
        top_k=args.top_k,
        vocabulary_ids=args.vocabulary_ids,
        threshold=args.threshold,
        standard_policy=pol,
        sleep_sec=sleep_sec,
        vocab_release=args.vocab_release,
    )
    print(f"저장 완료: {path}")


if __name__ == "__main__":
    main()
