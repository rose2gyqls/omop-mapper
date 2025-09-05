# SapBERT 의료 엔티티 임베딩 테스트 가이드

이 디렉토리에는 SapBERT를 사용하여 OMOP CONCEPT 데이터의 의료 엔티티 임베딩을 테스트하고 시각화하는 코드가 포함되어 있습니다.

## 🚀 빠른 시작

### 1. 의존성 설치
```bash
# 프로젝트 루트에서
poetry install
```

### 2. 빠른 테스트 (50개 샘플)
```bash
cd test
poetry run python run_sapbert_test.py --mode quick --sample_size 50
```

### 3. 특정 의료 용어 테스트
```bash
poetry run python run_sapbert_test.py --mode terms
```

### 4. 종합 테스트 (시각화 포함)
```bash
poetry run python run_sapbert_test.py --mode comprehensive --sample_size 1000
```

## 📁 파일 설명

### `sapbert_embedding_test.py`
메인 SapBERT 테스트 클래스가 포함된 파일입니다.

**주요 기능:**
- SapBERT 모델 로딩 및 임베딩 생성
- 유사성 분석 (코사인 유사도)
- 2D 시각화 (UMAP, t-SNE)
- 클러스터링 분석 (K-means)
- 인터랙티브 시각화 (Plotly)

### `run_sapbert_test.py`
간편한 실행을 위한 CLI 스크립트입니다.

**사용법:**
```bash
python run_sapbert_test.py [OPTIONS]

옵션:
  --csv_path PATH       CONCEPT.csv 파일 경로 (기본: ../data/CONCEPT.csv)
  --mode {quick,comprehensive,terms}  테스트 모드 (기본: quick)
  --sample_size INT     샘플 크기 (기본: 100)
  --output_dir PATH     결과 저장 디렉토리 (기본: ./sapbert_results)
```

### `sapbert_demo.ipynb`
Jupyter 노트북으로 인터랙티브하게 테스트할 수 있습니다.

**실행 방법:**
```bash
poetry run jupyter notebook sapbert_demo.ipynb
```

## 🧪 테스트 모드

### 1. Quick Mode (`--mode quick`)
- 빠른 성능 확인용
- 작은 샘플 크기 (50-200개 권장)
- 유사성 분석만 수행
- 실행 시간: 1-3분

### 2. Comprehensive Mode (`--mode comprehensive`)
- 완전한 분석 및 시각화
- 중간 샘플 크기 (500-2000개 권장)
- 모든 시각화 포함 (UMAP, t-SNE, 히트맵, 클러스터링)
- 결과를 HTML/PNG 파일로 저장
- 실행 시간: 5-15분

### 3. Terms Mode (`--mode terms`)
- 특정 의료 용어들로 테스트
- 미리 정의된 의료 용어 리스트 사용
- 용어 간 유사성 분석
- 실행 시간: 1분 미만

## 📊 결과 해석

### 유사성 점수
- **0.8-1.0**: 매우 높은 유사성 (거의 동일한 의미)
- **0.6-0.8**: 높은 유사성 (관련성이 높음)
- **0.4-0.6**: 중간 유사성 (어느 정도 관련성)
- **0.2-0.4**: 낮은 유사성 (약간의 관련성)
- **0.0-0.2**: 매우 낮은 유사성 (거의 무관)

### 성능 평가 기준
- **🟢 우수 (>0.7)**: 임베딩 품질이 매우 좋음
- **🟡 양호 (0.5-0.7)**: 임베딩 품질이 괜찮음
- **🔴 개선 필요 (<0.5)**: 임베딩 품질 향상 필요

## 🎨 생성되는 시각화

### 1. UMAP 2D 시각화
- 고차원 임베딩을 2D로 축소
- 도메인별 색상 구분
- 인터랙티브 Plotly 차트

### 2. t-SNE 2D 시각화
- 또 다른 차원 축소 방법
- 클러스터 구조 확인 가능

### 3. 유사성 히트맵
- 엔티티 간 유사성 매트릭스
- 색상으로 유사성 정도 표현

### 4. 클러스터링 분석
- K-means 클러스터링 결과
- 클러스터별 도메인 분포 분석

## 💡 사용 팁

### 샘플 크기 선택
- **개발/테스트**: 50-200개
- **품질 확인**: 500-1000개  
- **전체 분석**: 2000개 이상

### 메모리 사용량
- 1000개 샘플: ~2GB RAM
- 2000개 샘플: ~4GB RAM
- GPU 사용 시 VRAM도 고려

### 실행 시간
- CPU만 사용: 더 오래 걸림
- GPU 사용: 훨씬 빠름 (CUDA 지원 시)

## 🔧 문제 해결

### 메모리 부족 오류
```bash
# 배치 크기를 줄여서 실행
# sapbert_embedding_test.py에서 batch_size를 8 또는 4로 변경
```

### CUDA 오류
```bash
# CPU 모드로 강제 실행
export CUDA_VISIBLE_DEVICES=""
```

### 의존성 오류
```bash
# 의존성 재설치
poetry install --no-cache
```

## 📈 결과 예시

```
📊 테스트 결과 요약
==================================================
분석된 concept 수: 1,000
임베딩 차원: 768
전체 평균 유사성: 0.6234

도메인별 평균 유사성:
  Drug: 0.7123
  Condition: 0.6845
  Procedure: 0.5987
  Observation: 0.5234

💡 성능 평가:
   🟡 양호: 임베딩 품질이 괜찮습니다.

🔍 유사성 분석 예시:
1. 'Type 2 diabetes mellitus' (Condition)
   유사한 엔티티들:
   1. Diabetes mellitus - 유사도: 0.8956
   2. Non-insulin-dependent diabetes mellitus - 유사도: 0.8723
   3. Diabetic disorder - 유사도: 0.8234
```

## 🚀 고급 사용법

### 커스텀 의료 용어 테스트
```python
from sapbert_embedding_test import SapBERTEmbeddingTester

tester = SapBERTEmbeddingTester()

# 자신만의 의료 용어 리스트
custom_terms = ["your", "medical", "terms", "here"]
embeddings = tester.get_embeddings(custom_terms)

# 유사성 분석
similarity_results = tester.analyze_similarity(df, embeddings)
```

### 다른 SapBERT 모델 사용
```python
# 다국어 모델 사용
tester = SapBERTEmbeddingTester(
    model_name="cambridgeltl/SapBERT-XLMR-large"
)
```

## 📚 참고 자료

- [SapBERT 논문](https://aclanthology.org/2021.naacl-main.334/)
- [SapBERT GitHub](https://github.com/cambridgeltl/sapbert)
- [Hugging Face 모델](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext)

## 🤝 기여

버그 발견이나 개선 사항이 있으시면 이슈를 등록해주세요!
