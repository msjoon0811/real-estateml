"""
linear_regression.py
--------------------
Linear Regression 베이스라인 모델 (XGBoost 성능 비교 기준)

담당: 안우현

사용 방법
─────────
from src.models.linear_regression import train_linear, predict_linear

# 학습
model, metrics = train_linear(X_train, y_train, X_test, y_test)

# 예측
y_hat = predict_linear(model, X_new)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")          # GUI 없는 서버 환경 대응
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

# ──────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = "models/linear_regression.pkl"
REPORT_DIR         = "reports"


# ──────────────────────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    """경로의 부모 디렉터리가 없으면 생성."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """회귀 성능 지표 딕셔너리 반환."""
    return {
        "R2":   float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100),
    }


# ──────────────────────────────────────────────────────────────
# 학습 및 평가
# ──────────────────────────────────────────────────────────────

def train_linear(
    X_train,
    y_train,
    X_test,
    y_test,
    cv_folds: int = 5,
    save_path: Optional[str] = None,
    plot: bool = False,
) -> Tuple[LinearRegression, Dict[str, float]]:
    """
    Linear Regression 모델 학습 · 평가 · (선택) 저장 · (선택) 시각화.

    Parameters
    ----------
    X_train   : array-like (n_samples, n_features) — 학습 피처
    y_train   : array-like (n_samples,)            — 학습 타겟 (만원)
    X_test    : array-like                          — 검증 피처
    y_test    : array-like                          — 검증 타겟
    cv_folds  : K-Fold 교차 검증 횟수 (기본 5, 0이면 생략)
    save_path : 저장 경로 (None이면 저장 안 함)
    plot      : True면 예측 vs 실제 산점도를 reports/ 에 저장

    Returns
    -------
    model   : 학습된 LinearRegression
    metrics : {'R2', 'RMSE', 'MAE', 'MAPE'} 딕셔너리
    """
    # ── 학습 ───────────────────────────────────────
    model = LinearRegression()
    model.fit(X_train, y_train)

    # ── 검증 지표 ──────────────────────────────────
    y_pred   = model.predict(X_test)
    y_test_a = np.array(y_test)
    metrics  = _compute_metrics(y_test_a, y_pred)

    # ── 교차 검증 (선택) ───────────────────────────
    cv_rmse_mean = None
    if cv_folds > 1:
        kf       = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_r2    = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")
        cv_neg   = cross_val_score(model, X_train, y_train,
                                   cv=kf, scoring="neg_root_mean_squared_error")
        cv_rmse_mean = float(-cv_neg.mean())

    # ── 출력 ───────────────────────────────────────
    sep = "=" * 48
    print(sep)
    print("  [Linear Regression 베이스라인 결과]")
    print(f"  R²     : {metrics['R2']:.4f}")
    print(f"  RMSE   : {metrics['RMSE']:>12,.0f} 만원")
    print(f"  MAE    : {metrics['MAE']:>12,.0f} 만원")
    print(f"  MAPE   : {metrics['MAPE']:.2f} %")
    if cv_rmse_mean is not None:
        print(f"  CV-RMSE({cv_folds}fold): {cv_rmse_mean:>10,.0f} 만원")
    print(sep)

    # ── 계수 출력 (피처 수가 10 이하일 때만) ───────
    if hasattr(X_train, "shape") and X_train.shape[1] <= 10:
        print("  [회귀 계수]")
        for i, coef in enumerate(model.coef_):
            print(f"    feat[{i}] : {coef:+.4f}")
        print(f"  [절편]  : {model.intercept_:+.4f}")
        print(sep)

    # ── 시각화 ─────────────────────────────────────
    if plot:
        _plot_prediction(y_test_a, y_pred, label="LinearRegression",
                         fname="linear_pred_vs_actual.png")

    # ── 저장 ───────────────────────────────────────
    if save_path:
        save_linear(model, save_path)

    return model, metrics


# ──────────────────────────────────────────────────────────────
# 예측
# ──────────────────────────────────────────────────────────────

def predict_linear(model: LinearRegression, X) -> np.ndarray:
    """
    학습된 Linear Regression 모델로 가격 예측.

    Parameters
    ----------
    model : 학습된 LinearRegression
    X     : array-like — 예측할 피처 행렬

    Returns
    -------
    np.ndarray — 예측 거래금액 (만원)
    """
    return model.predict(X)


# ──────────────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────────────

def _plot_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label:  str  = "LinearRegression",
    fname:  str  = "linear_pred_vs_actual.png",
) -> None:
    """예측 vs 실제 산점도 저장 (reports/<fname>)."""
    os.makedirs(REPORT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 산점도
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.4, s=20, color="#4C72B0")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect fit")
    ax.set_xlabel("실제 거래금액 (만원)")
    ax.set_ylabel("예측 거래금액 (만원)")
    ax.set_title(f"{label} — 예측 vs 실제")
    ax.legend()

    # 잔차 히스토그램
    residuals = y_true - y_pred
    ax2 = axes[1]
    ax2.hist(residuals, bins=40, color="#55A868", edgecolor="white", alpha=0.85)
    ax2.axvline(0, color="red", linewidth=1.5, linestyle="--")
    ax2.set_xlabel("잔차 (만원)")
    ax2.set_ylabel("빈도")
    ax2.set_title("잔차 분포")

    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, fname)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  예측 시각화 저장 → {save_path}")


def plot_coefficients(
    model: LinearRegression,
    feature_names: list,
    fname: str = "linear_coefficients.png",
) -> None:
    """회귀 계수 막대 차트 저장."""
    os.makedirs(REPORT_DIR, exist_ok=True)

    coefs   = model.coef_
    indices = np.argsort(np.abs(coefs))[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_coefs = coefs[indices]
    colors = ["#C44E52" if c > 0 else "#4C72B0" for c in sorted_coefs]

    plt.figure(figsize=(10, max(4, len(feature_names) * 0.4)))
    plt.barh(sorted_names[::-1], sorted_coefs[::-1], color=colors[::-1])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("계수 값")
    plt.title("Linear Regression 회귀 계수 (|크기| 순)")
    plt.tight_layout()

    save_path = os.path.join(REPORT_DIR, fname)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  계수 차트 저장 → {save_path}")


# ──────────────────────────────────────────────────────────────
# 모델 저장 / 불러오기
# ──────────────────────────────────────────────────────────────

def save_linear(
    model: LinearRegression,
    path:  str = DEFAULT_MODEL_PATH,
) -> None:
    """모델을 joblib 포맷으로 저장."""
    _ensure_dir(path)
    joblib.dump(model, path)
    size_kb = Path(path).stat().st_size / 1024
    print(f"  Linear Regression 저장 완료 → {path}  ({size_kb:.1f} KB)")


def load_linear(path: str = DEFAULT_MODEL_PATH) -> LinearRegression:
    """저장된 모델 불러오기."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
    model = joblib.load(path)
    print(f"  Linear Regression 불러오기 완료 ← {path}")
    return model


# ──────────────────────────────────────────────────────────────
# 단독 실행 테스트
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    더미 데이터로 전체 파이프라인 동작 확인.

    실행:
        python -m src.models.linear_regression
    또는:
        python src/models/linear_regression.py
    """
    from sklearn.model_selection import train_test_split

    print("\n" + "=" * 48)
    print("  Linear Regression 단독 테스트 시작")
    print("=" * 48)

    rng = np.random.default_rng(42)
    n   = 800

    # 더미 부동산 피처
    area       = rng.uniform(40, 150, n)        # 전용면적(㎡)
    floor      = rng.integers(1, 30, n)         # 층수
    year       = rng.integers(1990, 2024, n)    # 건축연도
    households = rng.integers(50, 3000, n)      # 단지세대수
    dist       = rng.uniform(100, 5000, n)      # 역거리(m)

    X = np.column_stack([area, floor, year, households, dist])
    y = (area * 400 + floor * 200 - (2024 - year) * 50
         + households * 0.5 - dist * 0.8
         + rng.normal(0, 3000, n))              # 만원 단위

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    feature_names = ["전용면적", "층수", "건축연도", "단지세대수", "인근역거리"]

    model, metrics = train_linear(
        X_tr, y_tr, X_te, y_te,
        cv_folds=5,
        save_path="models/linear_regression.pkl",
        plot=True,
    )

    plot_coefficients(model, feature_names)

    # 단일 예측 확인
    sample = X_te[:3]
    preds  = predict_linear(model, sample)
    print("\n  [샘플 예측]")
    for i, (pred, actual) in enumerate(zip(preds, y_te[:3])):
        print(f"    샘플{i+1}: 예측={pred:,.0f}만원  실제={actual:,.0f}만원  "
              f"오차={actual-pred:+,.0f}만원")

    print("\n  테스트 완료 ✓")
