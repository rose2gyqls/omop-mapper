# OMOP CDM Entity Mapping API

![Project Image](https://img.shields.io/badge/OMOP-CDM-blue) ![Python](https://img.shields.io/badge/python-3.9+-blue.svg) ![Elasticsearch](https://img.shields.io/badge/Elasticsearch-9.0+-green.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg)

## Description

This project provides an intelligent entity mapping API for OMOP Common Data Model (CDM) concepts using hybrid search combining textual similarity and semantic similarity. The system leverages Elasticsearch for efficient concept retrieval and SapBERT embeddings for semantic understanding.

**Key Features:**
* **Hybrid Search**: Combines textual string matching with semantic similarity using SapBERT embeddings
* **OMOP CDM Compliant**: Full support for OMOP CDM concept mapping with standard/non-standard concept handling
* **Elasticsearch Integration**: High-performance search with concept embeddings stored in Elasticsearch
* **Multi-domain Support**: Handles various medical domains (Drug, Condition, Procedure, Measurement, etc.)
* **Real-time API**: Fast entity mapping with confidence scoring and relationship mapping

## Requirements

* **Python 3.9+**
* **Elasticsearch 9.0+**
* **Minimum 8GB VRAM GPU** (recommended for SapBERT model)
* **24GB RAM** (recommended for large-scale indexing)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/omop-mapper.git
cd omop-mapper
```

### 2. Set up the environment

#### Using Poetry (Recommended)
```bash
pip install poetry
poetry install
poetry shell
```

#### Using pip
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```bash
# Elasticsearch Configuration
ELASTICSEARCH_HOST=your-elasticsearch-host
ELASTICSEARCH_PORT=9200
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=your-password

# SapBERT Model Configuration
SAPBERT_MODEL=cambridgeltl/SapBERT-from-PubMedBERT-fulltext
GPU_DEVICE=0
```

## Data Preparation

### 1. Download OMOP CDM Concept Data

Download the CONCEPT.csv file from [ATHENA](https://athena.ohdsi.org/) and place it in the `data/` directory:

```
omop-mapper/
├── data/
│   └── CONCEPT.csv
```

## Elasticsearch Indexing

### 1. Basic Indexing

Index OMOP concepts with SapBERT embeddings:

```bash
python test/run_concept_indexing.py \
    --csv-path "./data/CONCEPT.csv" \
    --es-host "your-elasticsearch-host" \
    --es-port 9200 \
    --es-username "elastic" \
    --es-password "your-password" \
    --index-name "concepts" \
    --model-name "cambridgeltl/SapBERT-from-PubMedBERT-fulltext" \
    --batch-size 128 \
    --chunk-size 1000
```

**Arguments:**
* `--csv-path`: Path to the CONCEPT.csv file
* `--es-host`: Elasticsearch host address
* `--es-port`: Elasticsearch port (default: 9200)
* `--es-username`: Elasticsearch username
* `--es-password`: Elasticsearch password
* `--index-name`: Target index name (default: "concepts")
* `--model-name`: SapBERT model identifier
* `--batch-size`: Embedding generation batch size (default: 128)
* `--chunk-size`: Data processing chunk size (default: 1000)
* `--max-concepts`: Maximum number of concepts to process (optional)
* `--skip-concepts`: Number of concepts to skip (default: 0)
* `--gpu-device`: GPU device number (default: 0)

### 2. Test Indexing

For testing with a smaller dataset:

```bash
python test/run_concept_indexing.py \
    --test \
    --max-concepts 10000 \
    --batch-size 64
```

### 3. Domain-Specific Indexing

Index specific domains only:

```bash
python test/run_concept_indexing.py \
    --csv-path "./data/CONCEPT.csv" \
    --index-name "concepts-drug" \
    --domain-filter "Drug" \
    --max-concepts 50000
```

## Usage

### 1. Entity Mapping API

```python
from omop_mapper.entity_mapping_api import EntityMappingAPI, EntityInput, EntityTypeAPI

# Initialize the API
api = EntityMappingAPI()

# Check API health
health = api.health_check()
print(health)

# Map a single entity
entity = EntityInput(
    entity_name="diabetes",
    entity_type=EntityTypeAPI.CONDITION,
    confidence=1.0
)

results = api.map_entities([entity])
print(results)
```

### 2. Hybrid Search Testing

Test the hybrid search functionality:

```bash
python test/test_2_entity.py
```

### 3. Direct Elasticsearch Search

```python
from omop_mapper.elasticsearch_client import ElasticsearchClient

# Initialize client
es_client = ElasticsearchClient(
    host="your-elasticsearch-host",
    port=9200,
    username="elastic",
    password="your-password"
)

# Search concepts
results = es_client.search_concepts(
    query="myocardial infarction",
    domain_ids=["Condition"],
    standard_concept_only=True,
    limit=10
)

for result in results:
    print(f"{result['concept_name']} (ID: {result['concept_id']})")
```

## Configuration

### Elasticsearch Index Mapping

The system creates indices with the following structure:

```json
{
  "mappings": {
    "properties": {
      "concept_id": {"type": "keyword"},
      "concept_name": {
        "type": "text",
        "fields": {"keyword": {"type": "keyword"}}
      },
      "domain_id": {"type": "keyword"},
      "vocabulary_id": {"type": "keyword"},
      "concept_class_id": {"type": "keyword"},
      "standard_concept": {"type": "keyword"},
      "concept_code": {"type": "keyword"},
      "valid_start_date": {"type": "date", "format": "yyyyMMdd"},
      "valid_end_date": {"type": "date", "format": "yyyyMMdd"},
      "invalid_reason": {"type": "keyword"},
      "concept_embedding": {
        "type": "dense_vector",
        "dims": 768,
        "index": true,
        "similarity": "cosine"
      }
    }
  }
}
```
