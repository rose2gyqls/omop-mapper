# OMOP CONCEPT Elasticsearch 인덱싱

이 모듈은 OMOP CDM CONCEPT.csv 파일을 SapBERT 임베딩과 함께 Elasticsearch에 인덱싱하는 기능을 제공합니다.

## ✨ 두 가지 인덱싱 방식

### 🚀 방식 1: Eland + Ingest Pipeline (권장)
- **Eland를 통한 모델 배포**: SapBERT 모델을 Elasticsearch ML에 직접 배포
- **Ingest Pipeline**: 인덱싱 시 Elasticsearch에서 자동으로 임베딩 생성
- **로컬 부담 최소화**: GPU/CPU 리소스 부담 없음
- **확장성**: Elasticsearch 클러스터의 ML 리소스 활용

### 💻 방식 2: 로컬 임베딩 생성 (기존)
- **로컬 추론**: transformers 라이브러리로 로컬에서 임베딩 생성
- **높은 제어**: 배치 크기, 디바이스 등 세밀한 제어 가능
- **독립적**: Elasticsearch ML 기능 불필요

## 주요 기능

- **SapBERT 임베딩**: 의료 도메인에 특화된 SapBERT 모델을 사용하여 concept_name에 대한 고품질 임베딩 생성
- **대용량 데이터 처리**: 청크 단위 처리로 메모리 효율적인 대용량 CSV 파일 처리
- **Elasticsearch 인덱싱**: 코사인 유사도 기반 벡터 검색을 지원하는 Elasticsearch 인덱스 생성
- **배치 처리**: 효율적인 배치 처리로 빠른 인덱싱 성능 제공
- **Eland 통합**: ML 모델의 Elasticsearch 배포 및 관리

## 파일 구조

```
indexing/
├── __init__.py                    # 패키지 초기화
├── sapbert_embedder.py            # SapBERT 임베딩 생성기 (로컬 방식)
├── eland_model_manager.py         # Eland 모델 관리자 (권장 방식)
├── elasticsearch_indexer.py       # Elasticsearch 인덱서 (공통)
├── concept_data_processor.py      # CONCEPT CSV 데이터 처리기 (공통)
├── main_indexer.py               # 메인 스크립트 (로컬 방식)
├── main_indexer_eland.py         # 메인 스크립트 (Eland 방식) ⭐ 권장
├── test_eland_sapbert.py         # Eland + SapBERT 호환성 테스트
└── README.md                     # 이 파일
```

## 사용법

### 🚀 방식 1: Eland + Ingest Pipeline (권장)

#### 1-1. 기본 사용법
```bash
# ML 기능이 활성화된 Elasticsearch 실행 필요
python main_indexer_eland.py --csv-file /Users/rose/Desktop/omop-mapper/data/CONCEPT.csv --recreate-index
```

#### 1-2. 상세 옵션
```bash
python main_indexer_eland.py \
  --csv-file /Users/rose/Desktop/omop-mapper/data/CONCEPT.csv \
  --es-host localhost \
  --es-port 9200 \
  --index-name concepts \
  --es-model-id sapbert-pubmed \
  --pipeline-name concept-embedding-pipeline \
  --recreate-index \
  --chunk-size 1000 \
  --indexing-batch-size 500 \
  --log-level INFO
```

#### 1-3. 테스트용 소량 데이터
```bash
python main_indexer_eland.py \
  --csv-file /Users/rose/Desktop/omop-mapper/data/CONCEPT.csv \
  --max-rows 10000 \
  --recreate-index \
  --log-level DEBUG
```

### 💻 방식 2: 로컬 임베딩 생성 (기존)

#### 2-1. 기본 사용법
```bash
python main_indexer.py --csv-file /Users/rose/Desktop/omop-mapper/data/CONCEPT.csv --recreate-index
```

#### 2-2. 상세 옵션

```bash
python main_indexer.py \
  --csv-file /Users/rose/Desktop/omop-mapper/data/CONCEPT.csv \
  --es-host localhost \
  --es-port 9200 \
  --index-name concepts \
  --recreate-index \
  --chunk-size 1000 \
  --embedding-batch-size 128 \
  --indexing-batch-size 500 \
  --log-level INFO
```

## 🐳 Elasticsearch 실행 (ML 기능 포함)

Eland 방식을 사용하려면 ML 기능이 활성화된 Elasticsearch가 필요합니다:

```bash
# Docker로 ML 기능 포함 Elasticsearch 실행
docker run -d --name elasticsearch-ml \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "xpack.ml.enabled=true" \
  -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" \
  docker.elastic.co/elasticsearch/elasticsearch:9.1.0
```

## 명령행 옵션

### 필수 옵션
- `--csv-file`: CONCEPT.csv 파일 경로

### Elasticsearch 설정
- `--es-host`: Elasticsearch 호스트 (기본값: localhost)
- `--es-port`: Elasticsearch 포트 (기본값: 9200)
- `--es-username`: Elasticsearch 사용자명
- `--es-password`: Elasticsearch 비밀번호
- `--index-name`: 인덱스명 (기본값: concepts)

### SapBERT 설정
- `--sapbert-model`: SapBERT 모델명 (기본값: cambridgeltl/SapBERT-from-PubMedBERT-fulltext)

### 처리 설정
- `--chunk-size`: CSV 읽기 청크 크기 (기본값: 1000)
- `--embedding-batch-size`: 임베딩 생성 배치 크기 (기본값: 128)
- `--indexing-batch-size`: 인덱싱 배치 크기 (기본값: 500)
- `--max-rows`: 최대 처리할 행 수
- `--skip-rows`: 건너뛸 행 수 (기본값: 0)

### 기타 설정
- `--recreate-index`: 기존 인덱스 삭제 후 재생성
- `--log-level`: 로그 레벨 (DEBUG, INFO, WARNING, ERROR)
- `--log-file`: 로그 파일 경로

## Elasticsearch 인덱스 스키마

```json
{
  "mappings": {
    "properties": {
      "concept_id": { "type": "keyword" },
      "concept_name": {
        "type": "text",
        "fields": {
          "keyword": { "type": "keyword" }
        }
      },
      "domain_id": { "type": "keyword" },
      "vocabulary_id": { "type": "keyword" },
      "concept_class_id": { "type": "keyword" },
      "standard_concept": { "type": "keyword" },
      "concept_code": { "type": "keyword" },
      "valid_start_date": { "type": "date", "format": "yyyyMMdd" },
      "valid_end_date": { "type": "date", "format": "yyyyMMdd" },
      "invalid_reason": { "type": "keyword" },
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

## 개별 모듈 사용법

### SapBERT 임베딩 생성

```python
from indexing import SapBERTEmbedder

embedder = SapBERTEmbedder()
texts = ["covid-19", "high fever", "diabetes"]
embeddings = embedder.encode_texts(texts)
print(f"임베딩 형태: {embeddings.shape}")
```

### CONCEPT 데이터 처리

```python
from indexing import ConceptDataProcessor

processor = ConceptDataProcessor("/path/to/CONCEPT.csv")
total_rows = processor.get_total_rows()

for chunk_df in processor.read_concepts_in_chunks(chunk_size=1000):
    print(f"청크 크기: {len(chunk_df)}")
    # 데이터 처리...
```

### Elasticsearch 인덱싱

```python
from indexing import ConceptElasticsearchIndexer

indexer = ConceptElasticsearchIndexer(index_name="concepts")
indexer.create_index(delete_if_exists=True)

# 문서 인덱싱
documents = [{"concept_id": "123", "concept_name": "test", ...}]
indexer.index_concepts(documents)
```

## 성능 최적화 팁

1. **GPU 사용**: CUDA가 설치된 환경에서는 자동으로 GPU를 사용하여 임베딩 생성 속도를 향상시킵니다.

2. **배치 크기 조정**: 
   - `--embedding-batch-size`: GPU 메모리에 따라 조정 (128-512 권장)
   - `--indexing-batch-size`: Elasticsearch 성능에 따라 조정 (500-1000 권장)

3. **청크 크기 조정**:
   - `--chunk-size`: 메모리 사용량과 처리 속도의 균형을 고려하여 조정

4. **Elasticsearch 설정**:
   - 인덱싱 중에는 `refresh_interval`을 늘려서 성능 향상
   - `number_of_replicas`를 0으로 설정하여 인덱싱 속도 향상

## 문제 해결

### 메모리 부족
- `--chunk-size`와 `--embedding-batch-size`를 줄여보세요
- GPU 메모리가 부족한 경우 CPU 모드로 실행됩니다

### Elasticsearch 연결 오류
- Elasticsearch 서버가 실행 중인지 확인
- 호스트, 포트, 인증 정보를 확인

### CUDA 오류
- PyTorch와 CUDA 버전 호환성 확인
- GPU 메모리 부족 시 배치 크기 조정

## 로그 모니터링

인덱싱 진행 상황은 로그를 통해 모니터링할 수 있습니다:

```bash
# 로그 파일로 저장
python main_indexer.py --csv-file data.csv --log-file indexing.log

# 실시간 로그 확인
tail -f indexing.log
```

## 요구사항

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.21+
- Elasticsearch 8.0+
- pandas 2.0+
- numpy 1.26+

모든 의존성은 프로젝트 루트의 `pyproject.toml`에 정의되어 있습니다.
