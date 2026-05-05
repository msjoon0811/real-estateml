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
DATA_PATH      = "data/processed/integrated_dataset_v1_with_s1.csv"
OUTPUT_DIR     = "reports"
OUTPUT_PLOT    = os.path.join(OUTPUT_DIR, "peer_group_clusters.png")
OUTPUT_CSV     = "data/processed/integrated_dataset_v1_with_s1_s3.csv"
OUTLIER_THRESHOLD = 2.0  # Z-score 기준치


def find_elbow_k(sse_list, k_range):
    """기하학적 방식을 이용해 SSE 곡선에서 엘보우 포인트(최적의 K)를 찾습니다."""
    all_coords = np.vstack((k_range, sse_list)).T
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
    print("Peer Group 군집화 및 이상 거래 탐지 시작")
    print("="*60)

    # 1. 데이터 로드 및 전처리
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} 파일을 찾을 수 없습니다.")
        return

    print(f"[1/6] 데이터 불러오는 중... ({DATA_PATH})")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", low_memory=False)

    # 단지코드 결측치 필터링
    initial_rows = len(df)
    df_filtered = df[df['단지코드'].notna()].copy()
    print(f"  단지코드 notna 필터링: {initial_rows:,}건 -> {len(df_filtered):,}건")

    # 결측치 제거 (사용할 피처에 한해서)
    features = ['전용면적(㎡)', '건축년도', '시군구']
    df_filtered = df_filtered.dropna(subset=features + ['거래금액(만원)'])
    print(f"  피처 결측치 제거 후: {len(df_filtered):,}건")

    # 2. 피처 인코딩 및 스케일링
    print("[2/6] 피처 스케일링 및 원-핫 인코딩 진행 중...")
    numeric_features     = ['전용면적(㎡)', '건축년도']
    categorical_features = ['시군구']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
        ])

    X = preprocessor.fit_transform(df_filtered)

    # 3. 최적의 K값 탐색 (Elbow Method)
    print("[3/6] 최적의 클러스터 개수(K) 탐색 중... (K=2~15)")
    k_range = list(range(2, 16))
    sse = []

    X_sample = X[np.random.choice(X.shape[0], min(10000, X.shape[0]), replace=False)]

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_sample)
        sse.append(kmeans.inertia_)

    optimal_k = find_elbow_k(sse, k_range)
    print(f"  Elbow Method 최적 K: {optimal_k}개")

    # 4. K-Means 학습 및 Z-score 계산
    print(f"[4/6] K={optimal_k} 모델 학습 및 이상 거래(Z-score) 식별 중...")
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    df_filtered['S3_cluster'] = final_kmeans.fit_predict(X)

    cluster_stats = df_filtered.groupby('S3_cluster')['거래금액(만원)'].agg(['mean', 'std']).reset_index()
    df_filtered = df_filtered.merge(cluster_stats, on='S3_cluster')

    df_filtered['std'] = df_filtered['std'].replace(0, 1).fillna(1)

    df_filtered['S3_z_score']   = (df_filtered['거래금액(만원)'] - df_filtered['mean']) / df_filtered['std']
    df_filtered['S3_is_anomaly'] = (np.abs(df_filtered['S3_z_score']) > OUTLIER_THRESHOLD).astype(int)
    df_filtered = df_filtered.drop(columns=['mean', 'std'])

    outlier_count = df_filtered['S3_is_anomaly'].sum()
    print(f"  총 {len(df_filtered):,}건 중 이상 거래(|Z| > {OUTLIER_THRESHOLD}): {outlier_count:,}건 ({outlier_count/len(df_filtered)*100:.2f}%)")

    # 5. 원본 전체 데이터에 S3 컬럼 병합 후 CSV 저장
    print(f"[5/6] S3 점수 CSV 저장 중... ({OUTPUT_CSV})")
    s3_cols = df_filtered[['S3_cluster', 'S3_z_score', 'S3_is_anomaly']].copy()
    df_out  = df.copy()
    df_out[['S3_cluster', 'S3_z_score', 'S3_is_anomaly']] = np.nan
    df_out.loc[df_filtered.index, 'S3_cluster']    = s3_cols['S3_cluster'].values
    df_out.loc[df_filtered.index, 'S3_z_score']    = s3_cols['S3_z_score'].values
    df_out.loc[df_filtered.index, 'S3_is_anomaly'] = s3_cols['S3_is_anomaly'].values

    os.makedirs("data/processed", exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"  저장 완료: {OUTPUT_CSV} ({len(df_out):,}행)")

    # 6. 시각화
    print(f"[6/6] 차트 생성 및 저장 중... ({OUTPUT_PLOT})")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(k_range, sse, marker='o', linestyle='-', color='b')
    axes[0].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    axes[0].set_title('Elbow Method를 활용한 K값 탐색')
    axes[0].set_xlabel('클러스터 수 (K)')
    axes[0].set_ylabel('SSE (Inertia)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.5)

    sns.scatterplot(
        data=df_filtered,
        x='전용면적(㎡)',
        y='거래금액(만원)',
        hue=df_filtered['S3_is_anomaly'].astype(bool),
        palette={False: 'gray', True: 'red'},
        alpha=0.6,
        s=15,
        ax=axes[1]
    )
    axes[1].set_title(f'전용면적 대비 거래금액 산점도 (|Z| > {OUTLIER_THRESHOLD})')

    cluster_centers = df_filtered.groupby('S3_cluster')[['전용면적(㎡)', '거래금액(만원)']].mean()
    axes[1].scatter(
        cluster_centers['전용면적(㎡)'], cluster_centers['거래금액(만원)'],
        marker='X', color='blue', s=100, edgecolor='white', label='클러스터 중심'
    )

    handles, labels = axes[1].get_legend_handles_labels()
    new_labels = ['정상 거래', '이상 거래(저평가/고평가)', '클러스터 중심']
    if len(handles) >= 3:
        axes[1].legend(handles=handles[-3:], labels=new_labels)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close()

    print("\n완료!")
    print(f"  이미지: {OUTPUT_PLOT}")
    print(f"  데이터: {OUTPUT_CSV}")
    print("="*60)


if __name__ == "__main__":
    main()
