"""
OMOPHub REST API 클라이언트 (Basic / Semantic 검색).

문서: https://docs.omophub.com/api-reference/search/basic-search
     https://docs.omophub.com/api-reference/search/semantic-search
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal

DEFAULT_BASE_URL = "https://api.omophub.com/v1"


class OmopHubError(Exception):
    """API 오류 또는 비정상 응답."""


@dataclass
class SearchHit:
    concept_id: int | None
    concept_name: str | None
    score: float | None
    vocabulary_id: str | None
    domain_id: str | None
    concept_code: str | None
    standard_concept: str | None
    raw: dict[str, Any]


def _hit_from_row(row: dict[str, Any], score_key: str) -> SearchHit:
    return SearchHit(
        concept_id=row.get("concept_id"),
        concept_name=row.get("concept_name"),
        score=row.get(score_key),
        vocabulary_id=row.get("vocabulary_id"),
        domain_id=row.get("domain_id"),
        concept_code=row.get("concept_code"),
        standard_concept=row.get("standard_concept"),
        raw=row,
    )


def filter_standard_s_or_c(hits: list[SearchHit]) -> list[SearchHit]:
    """OMOP standard_concept 이 Standard(S) 또는 Classification(C) 인 항목만 유지."""
    out: list[SearchHit] = []
    for h in hits:
        sc = (h.standard_concept or "").strip().upper()
        if sc in ("S", "C"):
            out.append(h)
    return out


def filter_standard_s_only(hits: list[SearchHit]) -> list[SearchHit]:
    out: list[SearchHit] = []
    for h in hits:
        if (h.standard_concept or "").strip().upper() == "S":
            out.append(h)
    return out


class OmopHubClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_sec: float = 60.0,
    ):
        self.api_key = api_key or os.environ.get("OMOPHUB_API_KEY", "").strip()
        if not self.api_key:
            raise OmopHubError(
                "OMOPHUB_API_KEY 환경 변수가 없습니다. 대시보드에서 발급한 키를 설정하세요."
            )
        self.base_url = (base_url or os.environ.get("OMOPHUB_BASE_URL") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self.timeout_sec = timeout_sec

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    def _get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        q = {k: v for k, v in params.items() if v is not None and v != ""}
        url = f"{self.base_url}{path}?{urllib.parse.urlencode(q)}"
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                detail = str(e)
            raise OmopHubError(f"HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise OmopHubError(f"요청 실패: {e}") from e

        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            raise OmopHubError(f"JSON 파싱 실패: {body[:500]}") from e

        if not data.get("success", True) and "error" in data:
            raise OmopHubError(json.dumps(data.get("error", data), ensure_ascii=False))
        return data

    def search_basic(
        self,
        query: str,
        *,
        domain_ids: str | None = None,
        vocabulary_ids: str | None = None,
        page: int = 1,
        page_size: int = 10,
        vocab_release: str | None = None,
    ) -> list[SearchHit]:
        """키워드 기반 /v1/search/concepts"""
        params: dict[str, Any] = {
            "query": query,
            "page": page,
            "page_size": min(page_size, 1000),
        }
        if domain_ids:
            params["domain_ids"] = domain_ids
        if vocabulary_ids:
            params["vocabulary_ids"] = vocabulary_ids
        if vocab_release:
            params["vocab_release"] = vocab_release

        raw = self._get_json("/search/concepts", params)
        items = raw.get("data")
        if items is None:
            return []
        if isinstance(items, dict) and "concepts" in items:
            rows = items["concepts"]
        elif isinstance(items, list):
            rows = items
        else:
            rows = []

        hits: list[SearchHit] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            hits.append(_hit_from_row(row, "match_score"))
        return hits

    def search_semantic(
        self,
        query: str,
        *,
        domain_ids: str | None = None,
        vocabulary_ids: str | None = None,
        page: int = 1,
        page_size: int = 10,
        threshold: float | None = None,
        standard_concept: str | None = "S",
        vocab_release: str | None = None,
    ) -> list[SearchHit]:
        """의미 검색 /v1/concepts/semantic-search"""
        params: dict[str, Any] = {
            "query": query,
            "page": page,
            "page_size": min(page_size, 100),
        }
        if domain_ids:
            params["domain_ids"] = domain_ids
        if vocabulary_ids:
            params["vocabulary_ids"] = vocabulary_ids
        if threshold is not None:
            params["threshold"] = threshold
        if standard_concept:
            params["standard_concept"] = standard_concept
        if vocab_release:
            params["vocab_release"] = vocab_release

        raw = self._get_json("/concepts/semantic-search", params)
        inner = raw.get("data") or {}
        rows = inner.get("results") or []
        hits: list[SearchHit] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            hits.append(_hit_from_row(row, "similarity_score"))
        return hits

    def search(
        self,
        query: str,
        *,
        mode: Literal["semantic", "basic"] = "semantic",
        domain_id: str | None = None,
        vocabulary_ids: str | None = None,
        top_k: int = 5,
        threshold: float | None = None,
        standard_policy: Literal["s_or_c", "s_only", "none"] = "s_or_c",
        vocab_release: str | None = None,
    ) -> list[SearchHit]:
        """domain_id 하나를 OMOPHub query param 형식으로 전달.

        standard_policy:
        - s_or_c: API에는 standard 필터를 넓게 두거나 생략한 뒤, 응답에서 S·C만 남김 (기본).
        - s_only: semantic 시 API에 standard_concept=S (또는 응답에서 S만).
        - none: standard_concept 필터 없음.
        """
        dom = domain_id.strip() if domain_id else None
        voc = vocabulary_ids.strip() if vocabulary_ids else None

        # 필터 후 top_k개를 채우기 위해 한 번에 더 많이 요청
        fetch_n = top_k
        if standard_policy == "s_or_c":
            fetch_n = min(100 if mode == "semantic" else 1000, max(top_k * 8, 32))
        elif standard_policy == "s_only" and mode == "semantic":
            fetch_n = min(100, max(top_k, 1))

        api_standard: str | None
        if mode == "semantic":
            if standard_policy == "s_only":
                api_standard = "S"
            elif standard_policy == "s_or_c":
                api_standard = None
            else:
                api_standard = None
        else:
            api_standard = None

        if mode == "semantic":
            hits = self.search_semantic(
                query,
                domain_ids=dom,
                vocabulary_ids=voc,
                page_size=fetch_n,
                threshold=threshold,
                standard_concept=api_standard,
                vocab_release=vocab_release,
            )
        else:
            hits = self.search_basic(
                query,
                domain_ids=dom,
                vocabulary_ids=voc,
                page_size=fetch_n,
                vocab_release=vocab_release,
            )

        if standard_policy == "s_or_c":
            hits = filter_standard_s_or_c(hits)
        elif standard_policy == "s_only":
            if mode == "semantic" and api_standard == "S":
                pass
            else:
                hits = filter_standard_s_only(hits)

        return hits[:top_k]


def sleep_ratelimit(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)
