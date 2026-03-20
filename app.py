#!/usr/bin/env python3
"""Streamlit app for local OMOP entity mapping demos."""

import html
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from MapOMOP import DomainID, EntityInput, EntityMappingAPI, ElasticsearchClient, ScoringMode

load_dotenv(override=False)

APP_TITLE = "MapOMOP"
DEFAULT_MODEL = "gpt-5-mini-2025-08-07"
SUPPORTED_DOMAINS = [
    DomainID.CONDITION,
    DomainID.PROCEDURE,
    DomainID.DRUG,
    DomainID.OBSERVATION,
    DomainID.MEASUREMENT,
    DomainID.DEVICE,
]
DOMAIN_BY_LABEL = {"All domains": None, **{domain.value: domain for domain in SUPPORTED_DOMAINS}}


def _env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_runtime_config() -> dict[str, object]:
    return {
        "openai_api_key": _env("OPENAI_API_KEY"),
        "openai_model": _env("OPENAI_MODEL") or _env("LLM_MODEL") or DEFAULT_MODEL,
        "es_host": _env("ES_SERVER_HOST"),
        "es_port": _env_int("ES_SERVER_PORT", 9200),
        "es_username": _env("ES_SERVER_USERNAME"),
        "es_password": _env("ES_SERVER_PASSWORD"),
        "es_use_ssl": _env_bool("ES_USE_SSL", default=False),
    }


def get_missing_config(config: dict[str, object]) -> list[str]:
    env_to_config_key = {
        "OPENAI_API_KEY": "openai_api_key",
        "ES_SERVER_HOST": "es_host",
        "ES_SERVER_USERNAME": "es_username",
        "ES_SERVER_PASSWORD": "es_password",
    }
    missing = []
    for env_name, key in env_to_config_key.items():
        if config.get(key) is None:
            missing.append(env_name)
    return missing


@st.cache_data(ttl=30, show_spinner=False)
def get_es_health(
    host: Optional[str],
    port: int,
    username: Optional[str],
    password: Optional[str],
    use_ssl: bool,
) -> dict:
    client = ElasticsearchClient(
        host=host,
        port=port,
        username=username,
        password=password,
        use_ssl=use_ssl,
    )
    try:
        return client.health_check()
    finally:
        client.close()


@st.cache_resource(show_spinner=False)
def get_mapping_api(
    host: str,
    port: int,
    username: str,
    password: str,
    use_ssl: bool,
    model: str,
    openai_api_key: str,
) -> EntityMappingAPI:
    es_client = ElasticsearchClient(
        host=host,
        port=port,
        username=username,
        password=password,
        use_ssl=use_ssl,
    )
    return EntityMappingAPI(
        es_client=es_client,
        scoring_mode=ScoringMode.LLM,
        llm_provider="openai",
        llm_model=model or None,
        llm_api_key=openai_api_key,
    )


def build_results_frame(results: list) -> pd.DataFrame:
    rows = []
    for rank, result in enumerate(sorted(results, key=lambda item: item.mapping_score, reverse=True), start=1):
        rows.append(
            {
                "rank": rank,
                "concept_id": result.mapped_concept_id,
                "concept_name": result.mapped_concept_name,
                "domain": result.domain_id,
                "vocabulary": result.vocabulary_id,
                "concept_class": result.concept_class_id,
                "standard": result.standard_concept,
                "score": round(result.mapping_score, 4),
                "confidence": result.mapping_confidence,
                "method": result.mapping_method,
            }
        )
    return pd.DataFrame(rows)


def _rounded(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value, 4)


def _sql_null_display(value) -> str:
    """OMOP-style Missing: show SQL NULL instead of None/empty in UI tables."""
    if value is None:
        return "NULL"
    if isinstance(value, float) and pd.isna(value):
        return "NULL"
    s = str(value).strip()
    if s == "" or s.lower() in {"none", "nan"}:
        return "NULL"
    return s


_NULL_CELL_CSS = "color: #8b9eb0;"


def _null_cell_style(value: object) -> str:
    """Mimic Streamlit/None-style muted cells for SQL NULL and missing values."""
    try:
        if pd.isna(value):
            return _NULL_CELL_CSS
    except (TypeError, ValueError):
        pass
    if value == "NULL":
        return _NULL_CELL_CSS
    if isinstance(value, str) and value.strip().upper() == "NULL":
        return _NULL_CELL_CSS
    return ""


def _with_null_cell_style(df: pd.DataFrame):
    """Return DataFrame or pandas Styler (for NULL cell styling). Avoids pd.io.formats.style in type hints."""
    if df.empty:
        return df
    return df.style.map(_null_cell_style)


# OMOP CDM concept table schema (standard column order)
CONCEPT_SCHEMA = [
    "concept_id",
    "concept_name",
    "domain_id",
    "vocabulary_id",
    "concept_class_id",
    "standard_concept",
    "concept_code",
    "valid_start_date",
    "valid_end_date",
    "invalid_reason",
]


def build_result_details_frame(best_result) -> pd.DataFrame:
    """Build Best mapping details table (OMOP CDM concept schema)."""
    row = {
        "concept_id": best_result.mapped_concept_id,
        "concept_name": best_result.mapped_concept_name,
        "domain_id": best_result.domain_id,
        "vocabulary_id": best_result.vocabulary_id,
        "concept_class_id": best_result.concept_class_id,
        "standard_concept": _sql_null_display(best_result.standard_concept),
        "concept_code": best_result.concept_code,
        "valid_start_date": best_result.valid_start_date,
        "valid_end_date": best_result.valid_end_date,
        "invalid_reason": _sql_null_display(best_result.invalid_reason),
    }
    return pd.DataFrame([row])[CONCEPT_SCHEMA]


def build_stage1_frame(stage1_candidates: list[dict]) -> pd.DataFrame:
    """Stage 1 table: concept_id, concept_name, stage_1_score, match_type, + concept schema."""
    rows = []
    for candidate in stage1_candidates:
        score = candidate.get("elasticsearch_score") or candidate.get("_score", 0.0)
        match_type = candidate.get("search_type", "unknown")
        row = {
            "concept_id": candidate.get("concept_id", ""),
            "concept_name": candidate.get("concept_name", ""),
            "similarity_score": _rounded(float(score)) if score is not None else None,
            "match_type": match_type,
            "domain_id": candidate.get("domain_id", ""),
            "vocabulary_id": candidate.get("vocabulary_id", ""),
            "concept_class_id": candidate.get("concept_class_id", ""),
            "standard_concept": _sql_null_display(candidate.get("standard_concept")),
            "concept_code": candidate.get("concept_code", ""),
            "valid_start_date": candidate.get("valid_start_date"),
            "valid_end_date": candidate.get("valid_end_date"),
            "invalid_reason": _sql_null_display(candidate.get("invalid_reason")),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_stage3_frame(stage3_candidates: list[dict]) -> pd.DataFrame:
    """Stage 3 table: concept_id, concept_name, llm_reasoning, match_type, + concept schema."""
    rows = []
    sorted_candidates = sorted(
        stage3_candidates,
        key=lambda item: item.get("final_score", 0.0) or 0.0,
        reverse=True,
    )
    for candidate in sorted_candidates[:3]:
        llm_reasoning = candidate.get("llm_reasoning") or ""
        match_type = (
            "logic-based"
            if (candidate.get("llm_reasoning") or candidate.get("llm_score") is not None)
            else "similarity"
        )
        row = {
            "concept_id": candidate.get("concept_id", ""),
            "concept_name": candidate.get("concept_name", ""),
            "llm_reasoning": llm_reasoning,
            "match_type": match_type,
            "domain_id": candidate.get("domain_id", ""),
            "vocabulary_id": candidate.get("vocabulary_id", ""),
            "concept_class_id": candidate.get("concept_class_id", ""),
            "standard_concept": _sql_null_display(candidate.get("standard_concept")),
            "concept_code": candidate.get("concept_code", ""),
            "valid_start_date": candidate.get("valid_start_date"),
            "valid_end_date": candidate.get("valid_end_date"),
            "invalid_reason": _sql_null_display(candidate.get("invalid_reason")),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def render_best_mapping_card(best_result) -> None:
    concept_id = html.escape(str(best_result.mapped_concept_id or "-"))
    domain_id = html.escape(str(best_result.domain_id or "-"))
    standard = html.escape(str(best_result.standard_concept or "-"))
    st.markdown(
        f"""
        <section class="best-card">
            <p class="section-label">Best mapping</p>
            <h2 class="best-name">{html.escape(best_result.mapped_concept_name or "-")}</h2>
            <div class="pill-row">
                <span class="pill">Concept ID : {concept_id}</span>
                <span class="pill">Domain ID : {domain_id}</span>
                <span class="pill">Standard : {standard}</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_status_card(title: str, state: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="status-card">
            <p class="status-label">{title}</p>
            <p class="status-state">{state}</p>
            <p class="status-detail">{detail}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_llm_card(config: dict) -> None:
    """LLM status card: LLM ✅ / LLM ❌, model and source."""
    has_key = bool(config.get("openai_api_key"))
    title = "LLM ✅" if has_key else "LLM ❌"
    model = html.escape(str(config.get("openai_model", "-")))
    st.markdown(
        f"""
        <div class="status-card">
            <p class="status-label">{title}</p>
            <div class="status-rows">
                <div class="status-row"><span class="status-row-label">Model</span><span class="status-row-value">{model}</span></div>
                <div class="status-row"><span class="status-row-label">Source</span><span class="status-row-value">OPENAI_API_KEY</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_vocabulary_card() -> None:
    """VOCABULARY + linked 🔗; VERSION and TOTAL TERMS rows like other status cards."""
    athena_url = "https://athena.ohdsi.org/search-terms/start"
    st.markdown(
        f"""
        <div class="status-card">
            <p class="status-label">VOCABULARY<a href="{athena_url}" target="_blank" rel="noopener noreferrer" class="status-link-icon" aria-label="OHDSI Athena vocabulary">🔗</a></p>
            <div class="status-rows">
                <div class="status-row"><span class="status-row-label">Version</span><span class="status-row-value">V20250827</span></div>
                <div class="status-row"><span class="status-row-label">Total terms</span><span class="status-row-value">13,433,716</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ontology_card() -> None:
    """ONTOLOGY 📖: SapBERT + Elasticsearch engine; OMOP linkage types."""
    st.markdown(
        """
        <div class="status-card">
            <p class="status-label">ONTOLOGY 📖</p>
            <div class="status-rows">
                <div class="status-row"><span class="status-row-label">Engine</span><span class="status-row-value">SapBERT, Elasticsearch</span></div>
                <div class="status-row"><span class="status-row-label">OMOP</span><span class="status-row-value">relationship, synonym</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title=APP_TITLE, page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #f2f7f9;
        --bg-secondary: #e7f0f4;
        --surface: rgba(255, 255, 255, 0.94);
        --surface-strong: #ffffff;
        --primary: #0c6a7a;
        --primary-deep: #084c59;
        --primary-soft: rgba(12, 106, 122, 0.1);
        --accent: #1aa3b8;
        --text: #13293d;
        --muted: #557086;
        --border: rgba(12, 106, 122, 0.16);
        --shadow: rgba(19, 41, 61, 0.1);
        --input-placeholder: #7c96a8;
        --alert-text: #102a43;
        --status-caps-spacing: 0.055em;
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --bg: #08141c;
            --bg-secondary: #0d1e29;
            --surface: rgba(17, 33, 45, 0.96);
            --surface-strong: #132734;
            --primary: #5fc9d7;
            --primary-deep: #2c95a4;
            --primary-soft: rgba(95, 201, 215, 0.12);
            --accent: #8de3ee;
            --text: #e7f3f8;
            --muted: #9cb4c2;
            --border: rgba(95, 201, 215, 0.18);
            --shadow: rgba(0, 0, 0, 0.35);
            --input-placeholder: #6f8a9a;
            --alert-text: #e7f3f8;
            --status-caps-spacing: 0.055em;
        }
    }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: var(--bg);
    }
    .stApp {
        background:
            radial-gradient(circle at top right, var(--primary-soft), transparent 24rem),
            linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg) 44%, var(--bg) 100%);
        color: var(--text);
        font-family: "Manrope", "Helvetica Neue", sans-serif;
    }
    .stApp, .stApp p, .stApp label, .stApp span, .stApp li, .stApp div {
        color: var(--text);
    }
    .block-container {
        max-width: 1140px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        letter-spacing: -0.03em;
        color: var(--text);
        font-family: "Literata", Georgia, serif;
    }
    .hero {
        padding: 1.45rem 1.6rem 1.35rem 1.6rem;
        border: 1px solid var(--border);
        background: linear-gradient(145deg, var(--surface-strong), var(--surface));
        border-radius: 1.25rem;
        box-shadow: 0 18px 40px var(--shadow);
        margin-bottom: 1.2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute;
        inset: auto -2rem -2rem auto;
        width: 10rem;
        height: 10rem;
        border-radius: 999px;
        background: radial-gradient(circle, var(--primary-soft), transparent 70%);
    }
    .hero-title {
        font-size: 2.9rem;
        line-height: 1.05;
        font-weight: 800;
        margin: 0;
        color: var(--text);
        position: relative;
        z-index: 1;
    }
    .hero-tagline {
        font-family: "Manrope", "Helvetica Neue", sans-serif;
        font-size: 1.08rem;
        font-weight: 500;
        line-height: 1.45;
        color: var(--muted);
        margin: 0.75rem 0 0 0;
        max-width: none;
        white-space: nowrap;
        overflow-x: auto;
        overflow-y: hidden;
        position: relative;
        z-index: 1;
        -webkit-overflow-scrolling: touch;
    }
    .status-card {
        position: relative;
        min-height: 0;
        border-radius: 1.05rem;
        padding: 1.15rem 1.05rem 1.05rem;
        padding-top: 1.25rem;
        background: var(--surface);
        border: 1px solid var(--border);
        box-shadow: 0 14px 28px var(--shadow);
        backdrop-filter: blur(10px);
        overflow: hidden;
        font-family: "Manrope", "Helvetica Neue", sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    .status-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary), var(--accent));
    }
    .status-label {
        font-size: 0.9rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: var(--status-caps-spacing);
        color: var(--muted);
        margin: 0 0 0.5rem 0;
        padding: 0;
        border: none;
        line-height: 1.3;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.25em;
    }
    .status-state {
        font-size: 1.12rem;
        font-weight: 800;
        color: var(--text);
        line-height: 1.35;
        margin: 0 0 0.9rem 0;
    }
    .status-rows {
        display: table;
        width: 100%;
        table-layout: auto;
        border-collapse: separate;
        border-spacing: 0.4rem 0.5rem;
    }
    .status-row {
        display: table-row;
    }
    .status-row-label {
        display: table-cell;
        vertical-align: top;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: var(--status-caps-spacing);
        color: var(--muted);
        white-space: nowrap;
        text-align: left;
        width: 1%;
    }
    .status-row-value {
        display: table-cell;
        vertical-align: top;
        min-width: 0;
        font-size: 0.98rem;
        font-weight: 400;
        font-style: normal;
        letter-spacing: normal;
        line-height: 1.45;
        color: var(--text);
        text-align: left;
        white-space: normal;
        overflow-wrap: break-word;
        word-break: normal;
    }
    .status-row-link .status-row-value {
        font-weight: 400;
    }
    .status-about-lead {
        font-size: 0.92rem;
        font-weight: 700;
        margin: 0 0 0.65rem 0;
        line-height: 1.4;
    }
    .status-about-list {
        margin: 0;
        padding-left: 1.15rem;
        font-size: 0.92rem;
        color: var(--muted);
        line-height: 1.6;
    }
    .status-about-list li {
        margin-bottom: 0.35rem;
    }
    .status-about-list li:last-child {
        margin-bottom: 0;
    }
    .status-detail {
        font-size: 0.94rem;
        color: var(--muted);
        line-height: 1.55;
        margin: 0 0 0.42rem 0;
    }
    .status-detail:last-child {
        margin-bottom: 0;
    }
    .status-link {
        color: var(--primary);
        text-decoration: none;
        font-weight: 700;
    }
    .status-link:hover {
        text-decoration: underline;
    }
    .status-label .status-link-icon {
        color: inherit;
        font-weight: inherit;
        text-decoration: none;
    }
    .status-label .status-link-icon:hover {
        text-decoration: underline;
        opacity: 0.88;
    }
    .section-label {
        margin: 0 0 0.4rem 0;
        color: var(--primary);
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .section-title {
        margin: 0 0 0.3rem 0;
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text);
    }
    .section-copy {
        margin: 0 0 1rem 0;
        font-size: 0.98rem;
        color: var(--muted);
    }
    [data-testid="stForm"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 1.1rem;
        padding: 1.25rem 1.25rem 0.45rem 1.25rem;
        box-shadow: 0 14px 28px var(--shadow);
        position: relative;
        overflow: hidden;
        isolation: isolate;
    }
    [data-testid="stForm"]::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 0.28rem;
        background: linear-gradient(90deg, var(--primary), var(--accent));
        border-radius: 0;
        z-index: 0;
        pointer-events: none;
    }
    .best-card {
        background: linear-gradient(180deg, var(--surface-strong), var(--surface));
        border: 1px solid var(--border);
        border-radius: 1.2rem;
        padding: 1.3rem 1.35rem;
        box-shadow: 0 16px 34px var(--shadow);
        margin-bottom: 1rem;
    }
    .best-name {
        margin: 0;
        font-size: 2rem;
        color: var(--text);
    }
    .pill-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 0.9rem;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        padding: 0.42rem 0.78rem;
        border-radius: 999px;
        background: var(--primary-soft);
        border: 1px solid var(--border);
        color: var(--text);
        font-size: 0.92rem;
        font-weight: 700;
    }
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] div,
    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] > div,
    .stMarkdown code,
    .stCodeBlock,
    .stDataFrame,
    .stAlert {
        color: var(--text);
    }
    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] > div {
        background: var(--surface-strong);
        border-color: var(--border);
        min-height: 3rem;
        box-shadow: none;
    }
    .stTextInput label,
    .stSelectbox label {
        color: var(--text);
        font-weight: 700;
    }
    .stTextInput input::placeholder {
        color: var(--input-placeholder);
        opacity: 1;
    }
    .stTextInput input:focus,
    .stSelectbox div[data-baseweb="select"] > div:focus-within {
        border-color: var(--primary);
        box-shadow: 0 0 0 0.15rem var(--primary-soft);
    }
    .stButton button,
    .stFormSubmitButton button {
        background: linear-gradient(90deg, var(--primary), var(--accent));
        color: #ffffff;
        border: none;
        border-radius: 0.85rem;
        min-height: 3rem;
        font-weight: 700;
        box-shadow: 0 12px 24px var(--shadow);
    }
    .stButton button:hover,
    .stFormSubmitButton button:hover {
        background: var(--primary-deep);
        color: #ffffff;
    }
    .stAlert {
        background: var(--surface);
        border: 1px solid var(--border);
    }
    .stAlert p, .stAlert div {
        color: var(--alert-text);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 1rem;
        overflow: hidden;
        box-shadow: 0 14px 26px var(--shadow);
    }
    .stExpander {
        border-color: var(--border);
    }
    div[data-testid="stMarkdownContainer"] code {
        background: var(--primary-soft);
        color: var(--text);
        border-radius: 0.45rem;
        padding: 0.14rem 0.34rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

config = get_runtime_config()
missing_config = get_missing_config(config)
es_health = get_es_health(
    host=config["es_host"],
    port=config["es_port"],
    username=config["es_username"],
    password=config["es_password"],
    use_ssl=config["es_use_ssl"],
)

st.markdown(
    """
    <section class="hero">
        <h1 class="hero-title">MapOMOP</h1>
        <p class="hero-tagline">Understands clinical meaning, aligns with OMOP relationships, and selects the best standard OMOP concept.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

status_col1, status_col2, status_col3 = st.columns(3, gap="small")
with status_col1:
    render_vocabulary_card()
with status_col2:
    render_ontology_card()
with status_col3:
    render_llm_card(config)

if missing_config:
    st.error(
        "Missing required configuration: "
        + ", ".join(missing_config)
        + ". Copy .env.example to .env and fill in the missing values."
    )
elif es_health.get("status") != "connected":
    st.error(
        "Elasticsearch is not reachable with the current configuration. "
        + str(es_health.get("error", "Unknown connection error"))
    )

st.markdown(
    """
    <h2 class="section-title">Map a clinical term to OMOP</h2>
    <p class="section-copy">Enter a clinical term and select the target domain.</p>
    """,
    unsafe_allow_html=True,
)

with st.form("mapper_form"):
    input_col, domain_col = st.columns([1.5, 1.0], gap="large")
    with input_col:
        entity_name = st.text_input(
            "1. Clinical term",
            placeholder="e.g. myocardial ischemia",
            help="Enter the source clinical term you want to map into OMOP.",
        )
    with domain_col:
        selected_domain = st.selectbox(
            "2. Target domain",
            options=list(DOMAIN_BY_LABEL.keys()),
            index=1,
            help="Pick a specific OMOP domain or search across all supported domains.",
        )
    submitted = st.form_submit_button("Map to OMOP", use_container_width=True)

if submitted:
    if missing_config:
        st.error("Fill in the required .env values before running a mapping.")
    elif es_health.get("status") != "connected":
        st.error("Elasticsearch connection is not healthy yet. Fix the ES settings and retry.")
    elif not entity_name.strip():
        st.warning("Enter an entity name to map.")
    else:
        domain = DOMAIN_BY_LABEL[selected_domain]
        try:
            with st.spinner("Running MapOMOP"):
                api = get_mapping_api(
                    host=str(config["es_host"]),
                    port=int(config["es_port"]),
                    username=str(config["es_username"]),
                    password=str(config["es_password"]),
                    use_ssl=bool(config["es_use_ssl"]),
                    model=str(config["openai_model"]),
                    openai_api_key=str(config["openai_api_key"]),
                )
                results = api.map_entity(
                    EntityInput(entity_name=entity_name.strip(), domain_id=domain)
                )
        except Exception as exc:
            st.exception(exc)
        else:
            if not results:
                st.warning("No mapping candidates were returned for this input.")
            else:
                best_result = max(results, key=lambda item: item.mapping_score)
                stage1_candidates = getattr(api, "_last_stage1_candidates", []) or []
                stage3_candidates = getattr(api, "_last_rerank_candidates", []) or []
                stage1_frame = build_stage1_frame(stage1_candidates)
                stage3_frame = build_stage3_frame(stage3_candidates)
                detail_frame = build_result_details_frame(best_result)

                render_best_mapping_card(best_result)

                st.markdown("### Best mapping details")
                st.dataframe(
                    _with_null_cell_style(detail_frame),
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("### Mapping details")
                st.caption("How this match was selected")
                with st.expander(
                    f"Initial matches ({len(stage1_candidates)})",
                    expanded=False,
                ):
                    st.caption("Based on semantic, lexical similarity")
                    if not stage1_frame.empty:
                        st.dataframe(
                            _with_null_cell_style(stage1_frame),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No stage 1 candidates.")

                with st.expander(
                    "Final results (Top 3)",
                    expanded=True,
                ):
                    st.caption("Based on mapping logic and relationships")
                    if not stage3_frame.empty:
                        st.dataframe(
                            _with_null_cell_style(stage3_frame),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No stage 3 candidates.")
