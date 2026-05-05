import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 한글 폰트 설정 (Mac/Windows 대응)
if sys.platform == 'darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 상수 설정
DATA_PATH = "data/raw/integrated_dataset_v1.csv"
OUTPUT_DIR = "reports"
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "peer_group_clusters.png")
OUTLIER_THRESHOLD = 2.0  # Z-score 기준치

def find_elbow_k(sse_list, k_range):
    """기하학적 방식을 이용해 SSE 곡선에서 엘보우 포인트(최적의 K)를 찾습니다."""
    n_points = len(sse_list)
    all_coords = np.vstack((k_range, sse_list)).T
    # 첫점과 끝점을 잇는 직선 벡터
    line_vec = all_coords[-1] - all_coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    
    vec_from_first = all_coords - all_coords[0]
    scalar_proj = np.sum(vec_from_first * line_vec_norm, axis=1)
    vec_proj = np.outer(scalar_proj, line_vec_norm)
    vec_to_line = vec_from_first - vec_proj
    
    dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))
    best_idx = np.argmax(dist_to_line)
    return k_range[best_idx]

def main():
    print("="*60)
    print("🚀 Peer Group 군집화 및 이상 거래 탐지 시작")
    print("="*60)

    # 1. 데이터 로드 및 전처리
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} 파일을 찾을 수 없습니다.")
        return

    print(f"[1/5] 데이터 불러오는 중... ({DATA_PATH})")
    df = pd.read_csv(DATA_PATH)
    
    # 단지코드 결측치 필터링
    initial_rows = len(df)
    df = df[df['단지코드'].notna()].copy()
    filtered_rows = len(df)
    print(f"  👉 단지코드 notna 필터링 완료: {initial_rows:,}건 -> {filtered_rows:,}건")

    # 결측치 제거 (사용할 피처에 한해서)
    features = ['전용면적(㎡)', '건축년도', '시군구']
    df = df.dropna(subset=features + ['거래금액(만원)'])
    print(f"  👉 피처 결측치 제거 후 데이터 수: {len(df):,}건")

    # 2. 피처 인코딩 및 스케일링
    print("[2/5] 피처 스케일링 및 원-핫 인코딩 진행 중...")
    numeric_features = ['전용면적(㎡)', '건축년도']
    categorical_features = ['시군구']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
        ])

    X = preprocessor.fit_transform(df)

    # 3. 최적의 K값 탐색 (Elbow Method)
    print("[3/5] 최적의 클러스터 개수(K) 탐색 중... (K=2~15)")
    k_range = list(range(2, 16))
    sse = []
    
    # 샘플링을 통한 빠른 탐색 (선택적)
    # 데이터가 너무 크면 KMeans 학습에 시간이 걸리므로 샘플링하여 SSE 추정
    X_sample = X[np.random.choice(X.shape[0], min(10000, X.shape[0]), replace=False)]
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_sample)
        sse.append(kmeans.inertia_)
        
    optimal_k = find_elbow_k(sse, k_range)
    print(f"  👉 Elbow Method로 결정된 최적의 K: {optimal_k}개")

    # 4. K-Means 학습 및 Z-score 계산
    print(f"[4/5] K={optimal_k} 모델 학습 및 이상 거래(Z-score) 식별 중...")
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    df['Cluster'] = final_kmeans.fit_predict(X)

    # 각 클러스터별 거래금액 통계량(평균, 표준편차) 계산
    cluster_stats = df.groupby('Cluster')['거래금액(만원)'].agg(['mean', 'std']).reset_index()
    df = df.merge(cluster_stats, on='Cluster')
    
    # 단일 데이터 클러스터 등 std가 0이거나 NaN일 경우 대비
    df['std'] = df['std'].replace(0, 1).fillna(1)
    
    # Z-score 계산
    df['Z-score'] = (df['거래금액(만원)'] - df['mean']) / df['std']
    
    # 이상치 판별 (|Z| > threshold)
    df['Is_Outlier'] = np.abs(df['Z-score']) > OUTLIER_THRESHOLD
    outlier_count = df['Is_Outlier'].sum()
    
    print(f"  👉 총 {len(df):,}건 중 이상 거래(|Z| > {OUTLIER_THRESHOLD}) 발견: {outlier_count:,}건 ({outlier_count/len(df)*100:.2f}%)")

    # 5. 시각화 및 결과 저장
    print(f"[5/5] 차트 생성 및 저장 중... ({OUTPUT_PLOT})")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 왼쪽: Elbow 그래프
    axes[0].plot(k_range, sse, marker='o', linestyle='-', color='b')
    axes[0].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    axes[0].set_title('Elbow Method를 활용한 K값 탐색')
    axes[0].set_xlabel('클러스터 수 (K)')
    axes[0].set_ylabel('SSE (Inertia)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.5)

    # 오른쪽: 전용면적 vs 거래금액 산점도 (이상치 하이라이트)
    sns.scatterplot(
        data=df, 
        x='전용면적(㎡)', 
        y='거래금액(만원)', 
        hue='Is_Outlier',
        palette={False: 'gray', True: 'red'},
        alpha=0.6,
        s=15,
        ax=axes[1]
    )
    axes[1].set_title(f'전용면적 대비 거래금액 산점도 (|Z| > {OUTLIER_THRESHOLD})')
    
    # 클러스터 중심점 (시각적으로 참고용, 평균 거래금액과 면적 매칭)
    cluster_centers = df.groupby('Cluster')[['전용면적(㎡)', '거래금액(만원)']].mean()
    axes[1].scatter(
        cluster_centers['전용면적(㎡)'], cluster_centers['거래금액(만원)'], 
        marker='X', color='blue', s=100, edgecolor='white', label='클러스터 중심'
    )
    
    handles, labels = axes[1].get_legend_handles_labels()
    # 라벨 정리 (중심점 라벨 포함)
    new_labels = ['정상 거래', '이상 거래(저평가/고평가)', '클러스터 중심']
    if len(handles) >= 3:
        axes[1].legend(handles=handles[-3:], labels=new_labels)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close()

    print("\n✅ 모든 작업이 완료되었습니다!")
    print(f"   📊 생성된 이미지: {OUTPUT_PLOT}")
    print("="*60)

if __name__ == "__main__":
    main()
