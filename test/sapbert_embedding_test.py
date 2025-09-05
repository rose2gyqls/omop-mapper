"""
SapBERT를 사용한 의료 엔티티 임베딩 테스트 및 시각화
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import umap
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

class SapBERTEmbeddingTester:
    """SapBERT 임베딩 테스트 클래스"""
    
    def __init__(self, model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"):
        """
        SapBERT 모델 초기화
        
        Args:
            model_name: 사용할 SapBERT 모델명
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 중인 디바이스: {self.device}")
        
        # 토크나이저와 모델 로드
        print("SapBERT 모델을 로딩 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("모델 로딩 완료!")
        
    def load_concept_data(self, csv_path, sample_size=1000, random_state=42):
        """
        CONCEPT.csv 파일에서 샘플 데이터 로드
        
        Args:
            csv_path: CONCEPT.csv 파일 경로
            sample_size: 샘플링할 데이터 개수
            random_state: 랜덤 시드
        
        Returns:
            pandas.DataFrame: 샘플링된 concept 데이터
        """
        print(f"CONCEPT.csv 파일을 로딩 중... (샘플 크기: {sample_size})")
        
        # 전체 데이터 개수 확인
        total_lines = sum(1 for _ in open(csv_path)) - 1  # 헤더 제외
        print(f"전체 concept 개수: {total_lines:,}")
        
        # 랜덤 샘플링을 위한 인덱스 생성
        np.random.seed(random_state)
        skip_idx = sorted(np.random.choice(range(1, total_lines + 1), 
                                         size=total_lines - sample_size, 
                                         replace=False))
        
        # 데이터 로드
        df = pd.read_csv(csv_path, sep='\t', skiprows=skip_idx)
        
        # 기본적인 전처리
        df = df.dropna(subset=['concept_name'])
        df['concept_name'] = df['concept_name'].astype(str).str.strip()
        df = df[df['concept_name'] != '']
        
        print(f"로딩된 샘플 데이터: {len(df):,}개")
        print(f"도메인 분포:\n{df['domain_id'].value_counts()}")
        
        return df
    
    def get_embeddings(self, texts, batch_size=32, max_length=128):
        """
        텍스트 리스트를 SapBERT 임베딩으로 변환
        
        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 배치 크기
            max_length: 최대 토큰 길이
        
        Returns:
            numpy.ndarray: 임베딩 벡터들
        """
        all_embeddings = []
        
        print(f"{len(texts)}개 텍스트의 임베딩을 생성 중...")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i:i+batch_size]
                
                # 토크나이징
                encoded = self.tokenizer.batch_encode_plus(
                    batch_texts,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                
                # GPU로 이동
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # 임베딩 생성 (CLS 토큰 사용)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰
                
                all_embeddings.append(cls_embeddings.cpu().numpy())
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        print(f"임베딩 생성 완료! Shape: {embeddings.shape}")
        
        return embeddings
    
    def analyze_similarity(self, df, embeddings, top_k=5):
        """
        임베딩 기반 유사성 분석
        
        Args:
            df: concept 데이터프레임
            embeddings: 임베딩 벡터들
            top_k: 상위 k개 유사 엔티티 표시
        
        Returns:
            dict: 유사성 분석 결과
        """
        print("코사인 유사성 계산 중...")
        
        # 코사인 유사성 매트릭스 계산
        similarity_matrix = cosine_similarity(embeddings)
        
        # 각 도메인별 평균 유사성 계산
        domain_similarities = {}
        for domain in df['domain_id'].unique():
            domain_mask = df['domain_id'] == domain
            domain_indices = np.where(domain_mask)[0]
            
            if len(domain_indices) > 1:
                domain_sim_matrix = similarity_matrix[np.ix_(domain_indices, domain_indices)]
                # 대각선 제외 (자기 자신과의 유사성 제외)
                mask = ~np.eye(domain_sim_matrix.shape[0], dtype=bool)
                avg_similarity = domain_sim_matrix[mask].mean()
                domain_similarities[domain] = avg_similarity
        
        # 전체 평균 유사성 (대각선 제외)
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        overall_avg_similarity = similarity_matrix[mask].mean()
        
        # 몇 가지 예시 concept에 대한 유사한 엔티티 찾기
        sample_examples = []
        sample_indices = np.random.choice(len(df), size=min(5, len(df)), replace=False)
        
        for idx in sample_indices:
            concept_name = df.iloc[idx]['concept_name']
            domain = df.iloc[idx]['domain_id']
            
            # 자기 자신 제외하고 가장 유사한 엔티티들 찾기
            similarities = similarity_matrix[idx]
            similar_indices = np.argsort(similarities)[::-1][1:top_k+1]  # 자기 자신 제외
            
            similar_entities = []
            for sim_idx in similar_indices:
                similar_entities.append({
                    'name': df.iloc[sim_idx]['concept_name'],
                    'domain': df.iloc[sim_idx]['domain_id'],
                    'similarity': similarities[sim_idx]
                })
            
            sample_examples.append({
                'original': {'name': concept_name, 'domain': domain},
                'similar': similar_entities
            })
        
        results = {
            'overall_avg_similarity': overall_avg_similarity,
            'domain_similarities': domain_similarities,
            'sample_examples': sample_examples,
            'similarity_matrix': similarity_matrix
        }
        
        print(f"전체 평균 유사성: {overall_avg_similarity:.4f}")
        print("도메인별 평균 유사성:")
        for domain, sim in sorted(domain_similarities.items(), key=lambda x: x[1], reverse=True):
            print(f"  {domain}: {sim:.4f}")
        
        return results
    
    def visualize_embeddings_2d(self, df, embeddings, method='umap', save_path=None):
        """
        2D 임베딩 시각화 (UMAP 또는 t-SNE)
        
        Args:
            df: concept 데이터프레임
            embeddings: 임베딩 벡터들
            method: 차원 축소 방법 ('umap' 또는 'tsne')
            save_path: 저장할 파일 경로
        """
        print(f"{method.upper()}를 사용한 2D 시각화 생성 중...")
        
        # 차원 축소
        if method.lower() == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        else:
            raise ValueError("method는 'umap' 또는 'tsne'여야 합니다.")
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # 시각화 데이터 준비
        viz_df = df.copy()
        viz_df['x'] = embeddings_2d[:, 0]
        viz_df['y'] = embeddings_2d[:, 1]
        
        # Plotly 인터랙티브 시각화
        fig = px.scatter(
            viz_df, 
            x='x', 
            y='y', 
            color='domain_id',
            hover_data=['concept_name', 'vocabulary_id'],
            title=f'SapBERT 의료 엔티티 임베딩 시각화 ({method.upper()})',
            width=1000,
            height=700
        )
        
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(
            xaxis_title=f'{method.upper()} 1',
            yaxis_title=f'{method.upper()} 2',
            legend_title='도메인',
            font=dict(size=12)
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"시각화 저장됨: {save_path}")
        
        fig.show()
        
        return embeddings_2d, fig
    
    def visualize_similarity_heatmap(self, df, similarity_matrix, sample_size=50, save_path=None):
        """
        유사성 히트맵 시각화
        
        Args:
            df: concept 데이터프레임
            similarity_matrix: 유사성 매트릭스
            sample_size: 히트맵에 표시할 샘플 크기
            save_path: 저장할 파일 경로
        """
        print(f"유사성 히트맵 생성 중... (샘플 크기: {sample_size})")
        
        # 랜덤 샘플링
        sample_indices = np.random.choice(len(df), size=min(sample_size, len(df)), replace=False)
        sample_df = df.iloc[sample_indices].copy()
        sample_similarity = similarity_matrix[np.ix_(sample_indices, sample_indices)]
        
        # 라벨 생성 (concept_name을 짧게)
        labels = [name[:30] + '...' if len(name) > 30 else name 
                 for name in sample_df['concept_name'].values]
        
        # 히트맵 생성
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            sample_similarity,
            xticklabels=labels,
            yticklabels=labels,
            cmap='viridis',
            center=0.5,
            square=True,
            cbar_kws={'label': 'Cosine Similarity'}
        )
        
        plt.title('SapBERT 의료 엔티티 유사성 히트맵', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"히트맵 저장됨: {save_path}")
        
        plt.show()
    
    def cluster_analysis(self, df, embeddings, n_clusters=8, save_path=None):
        """
        클러스터링 분석
        
        Args:
            df: concept 데이터프레임
            embeddings: 임베딩 벡터들
            n_clusters: 클러스터 개수
            save_path: 저장할 파일 경로
        """
        print(f"K-means 클러스터링 수행 중... (클러스터 개수: {n_clusters})")
        
        # K-means 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # 클러스터별 도메인 분포 분석
        cluster_df = df.copy()
        cluster_df['cluster'] = cluster_labels
        
        # 클러스터별 통계
        cluster_stats = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = df[cluster_mask]
            
            domain_counts = cluster_data['domain_id'].value_counts()
            vocab_counts = cluster_data['vocabulary_id'].value_counts()
            
            cluster_stats.append({
                'cluster_id': cluster_id,
                'size': cluster_mask.sum(),
                'top_domain': domain_counts.index[0] if len(domain_counts) > 0 else 'N/A',
                'top_domain_ratio': domain_counts.iloc[0] / len(cluster_data) if len(domain_counts) > 0 else 0,
                'top_vocab': vocab_counts.index[0] if len(vocab_counts) > 0 else 'N/A',
                'sample_concepts': cluster_data['concept_name'].head(3).tolist()
            })
        
        # 클러스터 통계 출력
        print("\n클러스터 분석 결과:")
        for stat in cluster_stats:
            print(f"클러스터 {stat['cluster_id']} (크기: {stat['size']}):")
            print(f"  주요 도메인: {stat['top_domain']} ({stat['top_domain_ratio']:.2%})")
            print(f"  주요 어휘: {stat['top_vocab']}")
            print(f"  샘플 컨셉: {', '.join(stat['sample_concepts'][:2])}")
            print()
        
        # UMAP으로 클러스터 시각화
        reducer = umap.UMAP(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        
        viz_df = df.copy()
        viz_df['x'] = embeddings_2d[:, 0]
        viz_df['y'] = embeddings_2d[:, 1]
        viz_df['cluster'] = cluster_labels.astype(str)
        
        fig = px.scatter(
            viz_df,
            x='x',
            y='y',
            color='cluster',
            hover_data=['concept_name', 'domain_id'],
            title=f'SapBERT 임베딩 클러스터링 결과 (K={n_clusters})',
            width=1000,
            height=700
        )
        
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            legend_title='클러스터'
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"클러스터링 시각화 저장됨: {save_path}")
        
        fig.show()
        
        return cluster_labels, cluster_stats, fig
    
    def run_comprehensive_test(self, csv_path, sample_size=1000, output_dir="./sapbert_results"):
        """
        종합적인 SapBERT 임베딩 테스트 실행
        
        Args:
            csv_path: CONCEPT.csv 파일 경로
            sample_size: 테스트할 샘플 크기
            output_dir: 결과 저장 디렉토리
        """
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 60)
        print("SapBERT 의료 엔티티 임베딩 종합 테스트 시작")
        print("=" * 60)
        
        # 1. 데이터 로드
        df = self.load_concept_data(csv_path, sample_size=sample_size)
        
        # 2. 임베딩 생성
        concept_names = df['concept_name'].tolist()
        embeddings = self.get_embeddings(concept_names)
        
        # 3. 유사성 분석
        print("\n" + "="*50)
        print("유사성 분석 수행 중...")
        print("="*50)
        similarity_results = self.analyze_similarity(df, embeddings)
        
        # 유사성 예시 출력
        print("\n유사성 분석 예시:")
        for i, example in enumerate(similarity_results['sample_examples'][:3]):
            print(f"\n{i+1}. '{example['original']['name']}' ({example['original']['domain']})")
            print("   유사한 엔티티들:")
            for j, similar in enumerate(example['similar'][:3]):
                print(f"   {j+1}. {similar['name']} ({similar['domain']}) - 유사도: {similar['similarity']:.4f}")
        
        # 4. 2D 시각화 (UMAP)
        print("\n" + "="*50)
        print("UMAP 2D 시각화 생성 중...")
        print("="*50)
        embeddings_2d_umap, fig_umap = self.visualize_embeddings_2d(
            df, embeddings, method='umap', 
            save_path=os.path.join(output_dir, 'sapbert_embedding_umap.html')
        )
        
        # 5. t-SNE 시각화
        print("\n" + "="*50)
        print("t-SNE 2D 시각화 생성 중...")
        print("="*50)
        embeddings_2d_tsne, fig_tsne = self.visualize_embeddings_2d(
            df, embeddings, method='tsne',
            save_path=os.path.join(output_dir, 'sapbert_embedding_tsne.html')
        )
        
        # 6. 유사성 히트맵
        print("\n" + "="*50)
        print("유사성 히트맵 생성 중...")
        print("="*50)
        self.visualize_similarity_heatmap(
            df, similarity_results['similarity_matrix'], 
            sample_size=50,
            save_path=os.path.join(output_dir, 'similarity_heatmap.png')
        )
        
        # 7. 클러스터링 분석
        print("\n" + "="*50)
        print("클러스터링 분석 수행 중...")
        print("="*50)
        cluster_labels, cluster_stats, fig_cluster = self.cluster_analysis(
            df, embeddings, n_clusters=8,
            save_path=os.path.join(output_dir, 'clustering_analysis.html')
        )
        
        # 8. 결과 요약 저장
        summary = {
            'model_name': self.model_name,
            'sample_size': len(df),
            'embedding_dim': embeddings.shape[1],
            'overall_avg_similarity': similarity_results['overall_avg_similarity'],
            'domain_similarities': similarity_results['domain_similarities'],
            'cluster_stats': cluster_stats
        }
        
        # 요약 리포트 생성
        report_path = os.path.join(output_dir, 'sapbert_test_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("SapBERT 의료 엔티티 임베딩 테스트 리포트\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"모델: {summary['model_name']}\n")
            f.write(f"샘플 크기: {summary['sample_size']:,}\n")
            f.write(f"임베딩 차원: {summary['embedding_dim']}\n")
            f.write(f"전체 평균 유사성: {summary['overall_avg_similarity']:.4f}\n\n")
            
            f.write("도메인별 평균 유사성:\n")
            for domain, sim in sorted(summary['domain_similarities'].items(), 
                                    key=lambda x: x[1], reverse=True):
                f.write(f"  {domain}: {sim:.4f}\n")
            
            f.write("\n클러스터 분석 결과:\n")
            for stat in summary['cluster_stats']:
                f.write(f"클러스터 {stat['cluster_id']} (크기: {stat['size']}):\n")
                f.write(f"  주요 도메인: {stat['top_domain']} ({stat['top_domain_ratio']:.2%})\n")
                f.write(f"  주요 어휘: {stat['top_vocab']}\n\n")
        
        print(f"\n테스트 완료! 결과가 '{output_dir}' 디렉토리에 저장되었습니다.")
        print(f"리포트 파일: {report_path}")
        
        return {
            'df': df,
            'embeddings': embeddings,
            'similarity_results': similarity_results,
            'cluster_results': (cluster_labels, cluster_stats),
            'summary': summary
        }


def main():
    """메인 실행 함수"""
    # CONCEPT.csv 파일 경로 설정
    csv_path = "../data/CONCEPT.csv"
    
    # SapBERT 테스터 초기화
    tester = SapBERTEmbeddingTester()
    
    # 종합 테스트 실행
    results = tester.run_comprehensive_test(
        csv_path=csv_path,
        sample_size=1000,  # 테스트를 위해 1000개 샘플 사용
        output_dir="./sapbert_results"
    )
    
    print("\n🎉 SapBERT 임베딩 테스트가 성공적으로 완료되었습니다!")


if __name__ == "__main__":
    main()
