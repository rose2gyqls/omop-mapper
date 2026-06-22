# OMOP Mapper

OMOP Mapper maps free-text clinical entity names (conditions, drugs, measurements,
procedures, observations) to **OMOP CDM standard concepts**. It combines
Elasticsearch retrieval, SapBERT semantic embeddings, and an LLM reranking step,
and ships with a Streamlit demo, command-line tools, and an evaluation suite.

A hosted demo is available at **https://mapomop.onrender.com**.

## Method

Given an entity name and an optional target domain, mapping runs as a three-stage
pipeline (`src/MapOMOP/`):

1. **Candidate retrieval** (`stage1_candidate_retrieval.py`)
   Retrieves candidate concepts from Elasticsearch with three complementary
   strategies: lexical search (exact / phrase / fuzzy), semantic vector search over
   SapBERT embeddings, and a hybrid query that combines text, vector, and
   length similarity. Synonyms are searched and then resolved back to their original
   concepts.

2. **Standard concept collection** (`stage2_standard_collection.py`)
   Converts non-standard candidates to OMOP standard concepts by following
   concept relationships (e.g. `Maps to`), keeping only standard concepts.

3. **Hybrid scoring** (`stage3_hybrid_scoring.py`)
   Ranks the standard candidates and selects the best mapping. Three scoring modes
   are supported:
   - `llm` (default): LLM judgment without numeric similarity in the prompt
   - `llm_with_score`: LLM judgment with SapBERT semantic similarity in the prompt
   - `semantic`: SapBERT cosine similarity only

   LLM scoring is provider-agnostic via `LLMClient` (OpenAI and Together AI).

An optional LLM **validation** stage (`mapping_validation.py`, enabled with
`--validation`) can re-check the selected mapping.

## Requirements

- Python 3.10+ (3.11 recommended)
- An OpenAI API key (or another supported LLM provider)
- Access to the OMOP Elasticsearch indexes (host, port, credentials)

## Setup

```bash
git clone https://github.com/yourusername/omop-mapper.git
cd omop-mapper

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then fill in the values
```

Minimum `.env` values:

```bash
OPENAI_API_KEY=your-openai-api-key
ES_SERVER_HOST=your-es-host
ES_SERVER_PORT=9200
ES_SERVER_USERNAME=your-es-username
ES_SERVER_PASSWORD=your-es-password
ES_USE_SSL=false
```

## Usage

### Streamlit demo

```bash
streamlit run scripts/app.py
```

Enter an entity name and a target domain to see the top mapped concept and ranked
candidates.

### Python API

```python
from MapOMOP import EntityInput, EntityMappingAPI, DomainID

api = EntityMappingAPI()
entity = EntityInput(entity_name="myocardial ischemia", domain_id=DomainID.CONDITION)
results = api.map_entity(entity)
```

### Mapping CLI

Runs batch mapping over a dataset and writes `.json`, `.log`, and `.xlsx` to
`test_logs/`.

```bash
./scripts/run_mapping.sh snuh      # or: snomed
```

Common options:

| Option | Description |
| --- | --- |
| `-n, --sample-size N` | Limit the number of samples |
| `--sample-per-domain N` | Sample N entities per domain |
| `--random`, `--seed N` | Random sampling and seed |
| `-w, --workers N` | Parallel worker processes |
| `-r, --repeat N` | Repeat mapping N times (consistency check) |
| `--scoring {llm,llm_with_score,semantic}` | Stage-3 scoring mode |
| `--llm-provider {openai,together}`, `--llm-model` | LLM route override |
| `--validation` | Enable the LLM validation stage |

### Indexing CLI

Builds the Elasticsearch indexes from OMOP CDM data. Only needed if you maintain
your own indexes; the demo and mapping CLI just need access to an existing cluster.

```bash
./scripts/index.sh                 # local CSV (default)
./scripts/index.sh postgres        # PostgreSQL source
```

The PostgreSQL path additionally needs `PG_HOST`, `PG_PORT`, `PG_DBNAME`, `PG_USER`,
and `PG_PASSWORD` in `.env`.

### Evaluation

`eval/` contains scripts that build the run-20 mapping logs, consensus
evaluation workbooks, and baseline comparisons used in the study. They read and
write a working directory specified by `--base` or the `OMOP_EVAL_BASE`
environment variable:

```bash
export OMOP_EVAL_BASE=/path/to/eval-data
python eval/build_run20_logs.py
```

## Deployment

The app is deployed on [Render](https://render.com) and served at
**https://mapomop.onrender.com**. A [`render.yaml`](./render.yaml) Blueprint is
included: push the repository, create a Render Blueprint from it, and provide the
secrets (`OPENAI_API_KEY`, `ES_SERVER_HOST`, `ES_SERVER_USERNAME`,
`ES_SERVER_PASSWORD`). The start command is:

```bash
streamlit run scripts/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

## Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM scoring |
| `OPENAI_MODEL` | No | OpenAI model override |
| `ES_SERVER_HOST` | Yes | Elasticsearch host |
| `ES_SERVER_PORT` | No | Elasticsearch port (default `9200`) |
| `ES_SERVER_USERNAME` | Yes | Elasticsearch username |
| `ES_SERVER_PASSWORD` | Yes | Elasticsearch password |
| `ES_USE_SSL` | No | `true` or `false` (default `false`) |
| `PG_*` | Indexing only | PostgreSQL connection for the indexing CLI |
| `OMOP_EVAL_BASE` | Eval only | Working directory for `eval/` scripts |

Never commit `.env`. Keep Elasticsearch credentials out of tracked source files and
prefer read-only credentials for demo users.

## Project Structure

```text
omop-mapper/
├── src/MapOMOP/          # Core mapping package (3-stage pipeline + validation)
├── indexing/             # Elasticsearch index-building pipeline
├── scripts/              # CLIs and wrappers
│   ├── app.py            # Streamlit demo
│   ├── run_mapping.py    # Mapping CLI        (run_mapping.sh)
│   ├── run_indexing.py   # Indexing CLI       (index.sh)
│   ├── prepare_concept_small.py  # CONCEPT_SMALL.csv builder
│   └── mapping_common.py # Shared data loading / output helpers
├── eval/                 # Evaluation and analysis scripts
├── requirements.txt
├── render.yaml           # Render deployment blueprint
└── .env.example
```

## License

MIT
