# MapOMOP

Medical entity mapping to OMOP CDM standard concepts using a 3-stage hybrid search pipeline. Index OMOP CDM data into Elasticsearch, then map source entities (e.g., SNUH clinical terms, SNOMED) to standard concepts with lexical, semantic, and LLM-based scoring.

## Features

- **3-Stage Pipeline**: Candidate Retrieval → Standard Collection → LLM Scoring
- **Elasticsearch Indexing**: Local CSV or PostgreSQL with SapBERT embeddings
- **Multiple Scoring Modes**: LLM, semantic similarity, hybrid (ablation study)
- **Batch Mapping**: SNUH/SNOMED data sources with automatic preprocessing
- **Repeat Runs**: Consistency validation with incremental JSON/Excel output per run
- **Intermediate Results**: View partial results in Excel while mapping is in progress

## Prerequisites

- Python 3.9+
- Elasticsearch 8.0+
- GPU with 8GB+ VRAM (recommended for SapBERT embeddings)

## Installation

```bash
git clone https://github.com/yourusername/omop-mapper.git
cd omop-mapper

pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-api-key          # For LLM scoring
ES_SERVER_HOST=localhost
ES_SERVER_PORT=9200
ES_SERVER_USERNAME=elastic
ES_SERVER_PASSWORD=your-password
```

## Workflow Overview

1. **Index** — Load OMOP CDM data into Elasticsearch (concept, synonym, relationship).
2. **Map** — Run mapping on your data source (SNUH, SNOMED) with optional parallel workers and repeat runs.
3. **View** — Convert JSON results to Excel; use watch mode to view intermediate results in real time.

---

## Step 1: Indexing

Index OMOP CDM data into Elasticsearch. For local CSV, the script first prepares `CONCEPT_SMALL.csv` (CONCEPT + CONCEPT_SYNONYM merged), then indexes it along with relationship and synonym tables.

```bash
./scripts/index.sh
```

### Indexing Options

| Option | Description |
|-------|-------------|
| `--prepare-only` | Create CONCEPT_SMALL.csv only (no indexing) |
| `local_csv` \| `postgres` | Data source (default: local_csv) |
| `--data-folder` | Path to OMOP CDM CSV folder |
| `--tables` | Tables to index: `concept-small`, `synonym`, `relationship` |
| `--gpu` | GPU device (-1 for CPU) |
| `--no-embeddings` | Skip SapBERT embeddings |
| `--resume` | Resume from checkpoint after interruption |
| `--max-rows` | Limit rows (for testing) |
| `--bulk-delay` | Seconds between bulk requests (429 mitigation) |

---

## Step 2: Mapping

Run mapping on SNUH or SNOMED data. Outputs are written to `test_logs/` as JSON, LOG, and XLSX. With `--repeat N`, JSON and XLSX are saved incrementally after each run so you can view results without waiting for all runs to finish.

```bash
./scripts/run_mapping.sh snuh
```

Use `snomed` for SNOMED data. See options below.

### Mapping Options

| Option | Description |
|--------|-------------|
| `--sample-size`, `-n` | Max sample size |
| `--sample-per-domain N` | N entities per domain |
| `--random` | Random sampling |
| `--seed N` | Random seed (default: 42) |
| `--scoring` | `llm`, `llm_with_score`, `semantic`, `hybrid` |
| `--workers`, `-w` | Parallel workers (default: 1) |
| `--repeat`, `-r` | Repeat N times (consistency check) |

### Output Files

All outputs go to `test_logs/` with the same timestamp:

- `mapping_{snuh|snomed}_{timestamp}.json` — Raw results (incremental when `--repeat`)
- `mapping_{snuh|snomed}_{timestamp}.log` — Detailed logs
- `mapping_{snuh|snomed}_{timestamp}.xlsx` — Excel summary (incremental when `--repeat`)

---

## Step 3: View Results

Convert JSON or log files to Excel. Use `--watch` to refresh every 10 seconds and view intermediate results while mapping is still running.

```bash
python scripts/log_to_xlsx.py test_logs/mapping_snuh_20260304_123456.json
```

### View Options

| Option | Description |
|--------|--------------|
| `--watch` | Refresh every 10 seconds (for intermediate results) |
| `--csv` | CSV path for ground truth merge (log input only) |

---

## API Usage

```python
from MapOMOP import EntityMappingAPI, EntityInput, DomainID

api = EntityMappingAPI(scoring_mode="llm")

entity = EntityInput(
    entity_name="myocardial ischemia",
    domain_id=DomainID.CONDITION
)

results = api.map_entity(entity)

if results:
    best = max(results, key=lambda x: x.mapping_score)
    print(f"{best.mapped_concept_name} (ID: {best.mapped_concept_id})")
    print(f"Score: {best.mapping_score:.4f}")
```

---

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| **Stage 1** | Lexical + Semantic + Combined search over Elasticsearch |
| **Stage 2** | Non-standard → Standard via relationship transforms (alt_to, poss_eq, same_as, Marketed form of, etc.) and Maps to |
| **Stage 3** | LLM scoring |

---

## Project Structure

```
omop-mapper/
├── run_indexing.py           # Indexing CLI
├── run_mapping.py            # Mapping CLI
├── mapping_common.py         # Data sources, logging, JSON/XLSX output
├── prepare_concept_small.py  # CONCEPT_SMALL.csv preparation
├── indexing/
│   ├── unified_indexer.py
│   ├── elasticsearch_indexer.py
│   ├── sapbert_embedder.py
│   └── data_sources/
│       ├── local_csv.py
│       └── postgres.py
├── src/MapOMOP/
│   ├── entity_mapping_api.py
│   ├── elasticsearch_client.py
│   ├── mapping_validation.py
│   └── mapping_stages/
│       ├── stage1_candidate_retrieval.py
│       ├── stage2_standard_collection.py
│       └── stage3_hybrid_scoring.py
├── scripts/
│   ├── index.sh              # Indexing wrapper
│   ├── run_mapping.sh        # Mapping wrapper
│   └── log_to_xlsx.py        # JSON/Log → Excel
├── data/                     # CSV data sources
└── test_logs/                # Mapping outputs (JSON, LOG, XLSX)
```

---

## License

MIT
