"""
Autoencoder 기반 거래 이상 탐지 모델 (S1 점수 생성)

학습: 정상 거래(중개거래)만 사용 → 정상 패턴 학습
추론: 재구성 오차(Reconstruction Error)가 임계값 초과 시 이상 거래로 판정
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.utils.logger import get_logger

logger   = get_logger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

FEATURES = ["거래금액(만원)", "전용면적(㎡)", "층", "건축년도", "평당가(만원)"]


# ── 모델 정의 ────────────────────────────────────────────────────────────────

class ApartmentAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ── 데이터 준비 ───────────────────────────────────────────────────────────────

def load_and_prepare(csv_path: Path):
    """
    통합 데이터셋 로드 후 정상/전체 분리
    - 학습: 중개거래만 (정상 패턴 학습)
    - 추론: 전체 237,070건
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    df = df.dropna(subset=FEATURES)

    # 정상 거래 = 중개거래
    normal_df = df[df["거래유형"] == "중개거래"].copy()
    logger.info(f"전체: {len(df):,}건 / 학습용 정상 거래: {len(normal_df):,}건")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(normal_df[FEATURES].values).astype(np.float32)
    X_all   = scaler.transform(df[FEATURES].values).astype(np.float32)

    return X_train, X_all, df, scaler


# ── 학습 ─────────────────────────────────────────────────────────────────────

def train(model: ApartmentAutoencoder,
          X_train: np.ndarray,
          epochs: int = 50,
          batch_size: int = 256,
          lr: float = 1e-3) -> list[float]:

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset   = torch.utils.data.TensorDataset(torch.tensor(X_train))
    loader    = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            output = model(batch)
            loss   = criterion(output, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        losses.append(avg)
        if epoch % 10 == 0:
            logger.info(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg:.6f}")

    return losses


# ── 추론 ─────────────────────────────────────────────────────────────────────

def reconstruction_error(model: ApartmentAutoencoder,
                         X: np.ndarray) -> np.ndarray:
    """각 거래의 재구성 오차 계산 (클수록 이상 거래)"""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X)
        x_recon  = model(x_tensor)
        errors   = torch.mean((x_recon - x_tensor) ** 2, dim=1)
    return errors.numpy()


def compute_threshold(errors: np.ndarray, multiplier: float = 2.0) -> float:
    """임계값 = 평균 + multiplier * 표준편차"""
    return float(errors.mean() + multiplier * errors.std())


# ── 시각화 ────────────────────────────────────────────────────────────────────

def plot_error_distribution(train_errors: np.ndarray,
                            all_errors: np.ndarray,
                            threshold: float,
                            save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 학습 데이터 오차 분포
    axes[0].hist(train_errors, bins=100, color="steelblue", alpha=0.7, edgecolor="white")
    axes[0].axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"임계값: {threshold:.4f}")
    axes[0].set_title("정상 거래 재구성 오차 분포 (학습 데이터)", fontsize=13)
    axes[0].set_xlabel("재구성 오차 (MSE)")
    axes[0].set_ylabel("거래 수")
    axes[0].legend()

    # 전체 데이터 정상/이상 구분
    normal_mask  = all_errors <= threshold
    anomaly_mask = all_errors >  threshold
    axes[1].hist(all_errors[normal_mask],  bins=100, color="steelblue", alpha=0.7, label=f"정상 ({normal_mask.sum():,}건)")
    axes[1].hist(all_errors[anomaly_mask], bins=100, color="tomato",    alpha=0.7, label=f"이상 ({anomaly_mask.sum():,}건)")
    axes[1].axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"임계값: {threshold:.4f}")
    axes[1].set_title("전체 거래 이상 탐지 결과", fontsize=13)
    axes[1].set_xlabel("재구성 오차 (MSE)")
    axes[1].set_ylabel("거래 수")
    axes[1].legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"시각화 저장: {save_path}")


# ── 메인 실행 ─────────────────────────────────────────────────────────────────

def main():
    csv_path   = BASE_DIR / "data" / "processed" / "integrated_dataset_v1.csv"
    model_dir  = BASE_DIR / "models"
    report_dir = BASE_DIR / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. 데이터 준비
    X_train, X_all, df, scaler = load_and_prepare(csv_path)

    # 2. 모델 학습
    model = ApartmentAutoencoder(input_dim=len(FEATURES))
    logger.info("Autoencoder 학습 시작...")
    train(model, X_train, epochs=50, batch_size=256, lr=1e-3)

    # 3. 재구성 오차 계산
    train_errors = reconstruction_error(model, X_train)
    all_errors   = reconstruction_error(model, X_all)
    threshold    = compute_threshold(train_errors, multiplier=2.0)

    anomaly_count = int((all_errors > threshold).sum())
    anomaly_rate  = anomaly_count / len(all_errors) * 100
    logger.info(f"\n임계값: {threshold:.6f}")
    logger.info(f"이상 거래: {anomaly_count:,}건 ({anomaly_rate:.1f}%)")

    # 4. 결과 저장
    torch.save(model.state_dict(), model_dir / "autoencoder.pt")
    np.save(model_dir / "scaler_mean.npy", scaler.mean_)
    np.save(model_dir / "scaler_scale.npy", scaler.scale_)
    np.save(model_dir / "ae_threshold.npy", np.array([threshold]))
    logger.info("모델 저장 완료")

    # 5. S1 점수를 데이터셋에 추가 후 저장
    df["S1_ae_error"]   = all_errors
    df["S1_is_anomaly"] = (all_errors > threshold).astype(int)
    out_path = BASE_DIR / "data" / "processed" / "integrated_dataset_v1_with_s1.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"S1 점수 포함 데이터셋 저장: {out_path}")

    # 6. 시각화
    plot_error_distribution(
        train_errors, all_errors, threshold,
        report_dir / "autoencoder_error_dist.png"
    )

    # 7. 이상 거래 샘플 출력
    anomaly_df = df[df["S1_is_anomaly"] == 1][
        ["시군구", "단지명", "거래금액(만원)", "거래유형", "S1_ae_error"]
    ].sort_values("S1_ae_error", ascending=False).head(10)
    print("\n=== 이상 거래 상위 10건 ===")
    print(anomaly_df.to_string(index=False))


if __name__ == "__main__":
    main()
