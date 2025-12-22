# MapOMOP

Medical entity mapping to OMOP CDM standard concepts using a 3-stage hybrid search pipeline.

## Features

- **3-Stage Pipeline**: Candidate Retrieval → Standard Collection → Hybrid Scoring
- **Multiple Data Sources**: Local CSV, PostgreSQL, Athena API
- **SapBERT Embeddings**: Semantic similarity for medical concepts
- **LLM Validation**: OpenAI-based mapping validation

## Installation

```bash
# Clone
git clone https://github.com/yourusername/MapOMOP.git
cd MapOMOP

# Install dependencies
pip install -r requirements.txt

# Or use Poetry
poetry install
```

## Configuration

Create `.env` file:

```bash
OPENAI_API_KEY=your-api-key
ES_SERVER_HOST=localhost
ES_SERVER_PORT=9200
ES_SERVER_USERNAME=elastic
ES_SERVER_PASSWORD=your-password
```

## Indexing

```bash
# Local CSV
python run_unified_indexing.py local_csv --data-folder ./data/omop-cdm

# PostgreSQL
python run_unified_indexing.py postgres

# Options
--tables concept relationship synonym  # Tables to index
--gpu 0                                 # GPU device
--no-embeddings                         # Disable SapBERT
```

## Usage

```python
from MapOMOP import EntityMappingAPI, EntityInput, DomainID

api = EntityMappingAPI(scoring_mode="llm")

entity = EntityInput(
    entity_name="myocardial ischemia",
    domain_id=DomainID.CONDITION
)

results = api.map_entity(entity)

if results:
    best = results[0]
    print(f"{best.mapped_concept_name} (ID: {best.mapped_concept_id})")
    print(f"Score: {best.mapping_score:.4f}, Confidence: {best.mapping_confidence}")
```

## Project Structure

```
MapOMOP/
├── indexing/                    # Elasticsearch indexing
│   ├── data_sources/            # Data source adapters
│   │   ├── local_csv.py
│   │   ├── postgres.py
│   │   └── athena_api.py
│   ├── elasticsearch_indexer.py
│   ├── sapbert_embedder.py
│   └── unified_indexer.py
├── src/MapOMOP/             # Mapping API
│   ├── elasticsearch_client.py
│   ├── entity_mapping_api.py
│   ├── mapping_validation.py
│   └── mapping_stages/
│       ├── stage1_candidate_retrieval.py
│       ├── stage2_standard_collection.py
│       └── stage3_hybrid_scoring.py
├── run_unified_indexing.py      # Indexing CLI
└── data/omop-cdm/               # OMOP CDM files
```

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| **Stage 1** | Lexical + Semantic + Combined search |
| **Stage 2** | Non-standard → Standard conversion via "Maps to" |
| **Stage 3** | LLM or Hybrid scoring with validation |

## Requirements

- Python 3.9+
- Elasticsearch 8.0+
- GPU with 8GB+ VRAM (recommended)

## License

MIT
