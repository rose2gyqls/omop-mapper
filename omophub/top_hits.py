"""OMOPHub 배치 CSV의 top_hits_json 열 파싱."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


def domain_id_from_top_hits_json(raw: Any) -> str | None:
    """첫 번째 히트의 domain_id. 빈 배열·파싱 실패 시 None."""
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
    except (TypeError, ValueError):
        pass
    s = str(raw).strip()
    if not s or s == "[]":
        return None
    try:
        data = json.loads(s)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(data, list) or len(data) == 0:
        return None
    first = data[0]
    if not isinstance(first, dict):
        return None
    dom = first.get("domain_id")
    if dom is None:
        return None
    try:
        if pd.isna(dom):
            return None
    except (TypeError, ValueError):
        pass
    sdom = str(dom).strip()
    return sdom if sdom else None
