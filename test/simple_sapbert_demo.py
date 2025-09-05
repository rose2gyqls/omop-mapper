#!/usr/bin/env python3
"""
간단한 SapBERT 데모 스크립트
의료 용어들의 임베딩과 유사성을 빠르게 확인할 수 있습니다.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


def load_sapbert_model():
    """SapBERT 모델 로드"""
    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🧠 SapBERT 모델 로딩 중... (디바이스: {device})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    print("✅ 모델 로딩 완료!")
    
    return tokenizer, model, device


def get_embeddings(texts, tokenizer, model, device):
    """텍스트들을 임베딩으로 변환"""
    embeddings = []
    
    with torch.no_grad():
        for text in texts:
            # 토크나이징
            encoded = tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                max_length=128,
                return_tensors="pt"
            )
            
            # GPU로 이동
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # 임베딩 생성 ([CLS] 토큰 사용)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding[0])
    
    return np.array(embeddings)


def analyze_similarity(terms, embeddings):
    """유사성 분석 및 결과 출력"""
    similarity_matrix = cosine_similarity(embeddings)
    
    print(f"\n📊 {len(terms)}개 의료 용어의 유사성 분석 결과:")
    print("=" * 60)
    
    # 각 용어별 가장 유사한 용어들 찾기
    for i, term in enumerate(terms):
        similarities = similarity_matrix[i]
        # 자기 자신 제외하고 정렬
        similar_indices = np.argsort(similarities)[::-1][1:4]  # 상위 3개
        
        print(f"\n🔍 '{term}'과 가장 유사한 용어들:")
        for j, idx in enumerate(similar_indices):
            print(f"   {j+1}. {terms[idx]} (유사도: {similarities[idx]:.4f})")
    
    # 전체 평균 유사성
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    avg_similarity = similarity_matrix[mask].mean()
    print(f"\n📈 전체 평균 유사성: {avg_similarity:.4f}")
    
    # 가장 유사한/다른 쌍 찾기
    pairs = []
    for i in range(len(terms)):
        for j in range(i+1, len(terms)):
            pairs.append((terms[i], terms[j], similarity_matrix[i, j]))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n🔥 가장 유사한 쌍들:")
    for i, (term1, term2, sim) in enumerate(pairs[:3]):
        print(f"   {i+1}. {term1} ↔ {term2}: {sim:.4f}")
    
    print(f"\n❄️  가장 다른 쌍들:")
    for i, (term1, term2, sim) in enumerate(pairs[-3:]):
        print(f"   {i+1}. {term1} ↔ {term2}: {sim:.4f}")


def main():
    """메인 실행 함수"""
    print("🧬 SapBERT 의료 엔티티 임베딩 간단 데모")
    print("=" * 50)
    
    # 테스트할 의료 용어들
    medical_terms = [
        "diabetes mellitus",           # 당뇨병
        "type 2 diabetes",            # 2형 당뇨병  
        "hypertension",               # 고혈압
        "high blood pressure",        # 고혈압 (다른 표현)
        "myocardial infarction",      # 심근경색
        "heart attack",               # 심장마비
        "pneumonia",                  # 폐렴
        "lung infection",             # 폐감염
        "influenza",                  # 인플루엔자
        "flu",                        # 독감
        "covid-19",                   # 코로나19
        "coronavirus infection",       # 코로나바이러스 감염
        "headache",                   # 두통
        "migraine",                   # 편두통
        "fever",                      # 발열
        "high temperature",           # 고체온
        "cough",                      # 기침
        "chest pain",                 # 흉통
        "shortness of breath",        # 호흡곤란
        "difficulty breathing"        # 호흡 어려움
    ]
    
    print(f"📋 테스트 용어 수: {len(medical_terms)}")
    print(f"📝 용어 목록: {', '.join(medical_terms[:5])}...")
    
    try:
        # 1. 모델 로드
        tokenizer, model, device = load_sapbert_model()
        
        # 2. 임베딩 생성
        print(f"\n⚡ 임베딩 생성 중...")
        embeddings = get_embeddings(medical_terms, tokenizer, model, device)
        print(f"✅ 임베딩 생성 완료! 형태: {embeddings.shape}")
        
        # 3. 유사성 분석
        analyze_similarity(medical_terms, embeddings)
        
        # 4. 성능 평가
        similarity_matrix = cosine_similarity(embeddings)
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        avg_similarity = similarity_matrix[mask].mean()
        
        print(f"\n💡 SapBERT 성능 평가:")
        if avg_similarity > 0.7:
            print("   🟢 우수: 의료 용어 임베딩 품질이 매우 좋습니다!")
        elif avg_similarity > 0.5:
            print("   🟡 양호: 의료 용어 임베딩 품질이 괜찮습니다.")
        else:
            print("   🔴 개선 필요: 임베딩 품질 향상이 필요합니다.")
        
        print(f"\n🎉 데모 완료!")
        print(f"💬 더 자세한 분석을 원하시면 'python run_sapbert_test.py --mode comprehensive' 를 실행해보세요!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        print("💡 해결 방법:")
        print("   1. 인터넷 연결 확인 (모델 다운로드 필요)")
        print("   2. 충분한 메모리 확인 (최소 4GB RAM 권장)")
        print("   3. PyTorch와 transformers 라이브러리 설치 확인")


if __name__ == "__main__":
    main()
