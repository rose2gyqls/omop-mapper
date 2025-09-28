# OMOP CDM Entity Mapping API

![Project Image](https://img.shields.io/badge/OMOP-CDM-blue) ![Python](https://img.shields.io/badge/python-3.9+-blue.svg) ![Elasticsearch](https://img.shields.io/badge/Elasticsearch-9.0+-green.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg)

## Description

This project provides an intelligent entity mapping API for OMOP Common Data Model (CDM) concepts using a **3-stage hybrid search** pipeline that combines textual similarity and semantic similarity. The system leverages Elasticsearch for efficient concept retrieval and SapBERT embeddings for semantic understanding.

**Key Features:**
* **3-Stage Mapping Pipeline**: 
  1. Elasticsearch query for top 5 candidates
  2. Standard/Non-standard classification and candidate collection
  3. Hybrid scoring with concept embeddings
* **Semantic Understanding**: SapBERT embeddings for medical concept semantic similarity
* **OMOP CDM Compliant**: Full support for OMOP CDM concept mapping with standard/non-standard concept handling
* **Elasticsearch Integration**: High-performance search with concept embeddings stored in Elasticsearch

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
OMOP_ENABLE_EMBEDDING=1
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
python run_indexing.py \
    --csv-path "./data/CONCEPT.csv" \
    --es-host "your-elasticsearch-host" \
    --es-port 9200 \
    --es-username "elastic" \
    --es-password "your-password" \
    --index-name "concept" \
    --model-name "cambridgeltl/SapBERT-from-PubMedBERT-fulltext" \
    --batch-size 128 \
    --chunk-size 1000
```

### 2. Test Indexing

For testing with a smaller dataset:

```bash
python run_indexing.py \
    --csv-path "./data/CONCEPT.csv" \
    --max-concepts 10000 \
    --batch-size 64
```

## Usage

### 1. Entity Mapping API

```python
from omop_mapper.entity_mapping_api import EntityMappingAPI, EntityInput, DomainID

# Initialize the API
api = EntityMappingAPI()

# Check API health
health = api.health_check()
print(health)

# Map a single entity
entity = EntityInput(
    entity_name="myocardial ischemia",
    domain_id=DomainID.CONDITION,
    vocabulary_id="SNOMED"  # Optional
)

result = api.map_entity(entity)

if result:
    print(f"Mapped: {result.mapped_concept_name} (ID: {result.mapped_concept_id})")
    print(f"Score: {result.mapping_score:.4f}")
    print(f"Confidence: {result.mapping_confidence}")
    print(f"Method: {result.mapping_method}")
    
    # Alternative concepts
    for alt in result.alternative_concepts:
        print(f"Alternative: {alt['concept_name']} (Score: {alt['score']:.4f})")
```

### 2. 3-Stage Mapping Testing

Test the complete 3-stage mapping pipeline:

```bash
# Test with sample data
python test_entity_mapping_with_logging.py
```

**Test Output:**
```
📊 3단계 후보군 상세 정보:
   1. Myocardial ischemia (ID: 4186397)
      - 텍스트 유사도: 1.0000
      - 의미적 유사도: 1.0000
      - 최종 점수: 1.0000
      - Vocabulary: SNOMED

✅ 매핑 성공!
   - 매핑된 컨셉: Myocardial ischemia (ID: 4186397)
   - 매핑 점수: 1.0000
   - 매핑 신뢰도: very_high
   - 매핑 방법: direct_standard
   - Vocabulary: SNOMED
```

## 3-Stage Mapping Pipeline

### Stage 1: Elasticsearch Query
- Searches top 5 candidates using text-based queries
- Combines exact phrase matching with token-based matching
- Returns candidates regardless of standard/non-standard status

### Stage 2: Standard Candidate Collection
- **Standard Concepts (S/C)**: Added directly to candidate pool
- **Non-standard Concepts**: Maps to standard concepts via "Maps to" relationships
- **Deduplication**: Removes duplicate concepts based on concept_id and concept_name
- **Relationship Resolution**: Uses concept_relationship index for mapping chains

### Stage 3: Hybrid Scoring
- **Text Similarity**: N-gram 3 based Jaccard similarity (40% weight)
- **Semantic Similarity**: SapBERT embedding cosine similarity (60% weight)
- **Final Ranking**: Sorts by combined hybrid score
- **Confidence Levels**: very_high (≥0.95), high (≥0.85), medium (≥0.70), low (≥0.50), very_low (<0.50)

### Mapping Result Structure

```python
@dataclass
class MappingResult:
    source_entity: EntityInput
    mapped_concept_id: str
    mapped_concept_name: str
    domain_id: str
    vocabulary_id: str
    concept_class_id: str
    standard_concept: str
    concept_code: str
    concept_embedding: List[float]
    mapping_score: float
    mapping_confidence: str  # very_high, high, medium, low, very_low
    mapping_method: str  # direct_standard, non_standard_to_standard
    alternative_concepts: List[Dict[str, Any]]
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