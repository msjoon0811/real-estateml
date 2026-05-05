"""
xgboost_regressor.py
--------------------
XGBoost 기반 아파트 실거래가 예측 + 괴리율 계산

담당: 안우현

사용 방법
─────────
from src.models.xgboost_regressor import train_xgboost, compute_deviation_score

model, metrics = train_xgboost(X_train, y_train, X_test, y_test)
deviation      = compute_deviation_score(model, X_test, y_test)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ──────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = "models/xgboost_regressor.pkl"
REPORT_DIR         = "reports"

# 괴리율 임계값
DEVIATION_WARNING  = 0.20   # ±20%  이상 → 주의
DEVIATION_CRITICAL = 0.40   # ±40%  이상 → 고위험


# ──────────────────────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2":   float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100),
    }


# ──────────────────────────────────────────────────────────────
# 학습
# ──────────────────────────────────────────────────────────────

def train_xgboost(
    X_train,
    y_train,
    X_test,
    y_test,
    n_estimators:   int   = 300,
    learning_rate:  float = 0.05,
    max_depth:      int   = 6,
    subsample:      float = 0.8,
    colsample:      float = 0.8,
    early_stopping: int   = 30,
    save_path:      Optional[str] = None,
    plot:           bool  = False,
    feature_names:  Optional[List[str]] = None,
) -> Tuple[XGBRegressor, Dict[str, float]]:
    """
    XGBoost Regressor 학습 · 평가 · (선택) 저장 · (선택) 시각화.

    Parameters
    ----------
    X_train / y_train   : 학습 데이터
    X_test  / y_test    : 검증 데이터
    n_estimators        : 트리 수 (기본 300)
    learning_rate       : 학습률 (기본 0.05)
    max_depth           : 최대 트리 깊이 (기본 6)
    subsample           : 행 샘플링 비율 (기본 0.8)
    colsample           : 열 샘플링 비율 (기본 0.8)
    early_stopping      : 조기 종료 라운드 (기본 30, 0이면 비활성)
    save_path           : 모델 저장 경로 (None이면 저장 안 함)
    plot                : True면 학습 곡선 + 예측 차트 저장
    feature_names       : 피처 이름 리스트 (시각화 시 표시)

    Returns
    -------
    model   : 학습된 XGBRegressor
    metrics : {'R2', 'RMSE', 'MAE', 'MAPE'} 딕셔너리
    """
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample,
        random_state=42,
        tree_method="hist",        # CPU 히스토그램 기반 (빠름)
        eval_metric="rmse",
        early_stopping_rounds=early_stopping if early_stopping > 0 else None,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50,
    )

    # ── 지표 계산 ──────────────────────────────────
    y_test_a = np.array(y_test)
    y_pred   = model.predict(X_test)
    metrics  = _compute_metrics(y_test_a, y_pred)

    sep = "=" * 48
    print(sep)
    print("  [XGBoost 최종 결과]")
    print(f"  Best iteration : {model.best_iteration}")
    print(f"  R²     : {metrics['R2']:.4f}")
    print(f"  RMSE   : {metrics['RMSE']:>12,.0f} 만원")
    print(f"  MAE    : {metrics['MAE']:>12,.0f} 만원")
    print(f"  MAPE   : {metrics['MAPE']:.2f} %")
    print(sep)

    # ── 시각화 ─────────────────────────────────────
    if plot:
        _plot_learning_curve(model)
        _plot_prediction(y_test_a, y_pred, label="XGBoost")
        if feature_names:
            plot_feature_importance(model, feature_names)

    # ── 저장 ───────────────────────────────────────
    if save_path:
        save_xgboost(model, save_path)

    return model, metrics


# ──────────────────────────────────────────────────────────────
# 괴리율 계산
# ──────────────────────────────────────────────────────────────

def compute_deviation_score(
    model,
    X,
    y_actual,
    verbose: bool = False,
) -> np.ndarray:
    """
    괴리율 계산: (실제가 - 예측가) / 예측가

    해석
    ----
    음수(-) : 실제 < 예측  →  저평가 (허위 낮은 신고 의심)
    양수(+) : 실제 > 예측  →  고평가 (담합·기획 거래 의심)

    임계값 (DEVIATION_*)
    --------------------
    |괴리율| < 0.20  → 정상
    |괴리율| 0.20~0.40 → 주의
    |괴리율| ≥ 0.40  → 고위험

    Parameters
    ----------
    model    : 학습된 XGBRegressor
    X        : 피처 행렬
    y_actual : 실제 거래금액 배열 (만원)
    verbose  : True면 요약 통계 출력

    Returns
    -------
    np.ndarray  — 샘플별 괴리율 (소수, 음수/양수 가능)
    """
    y_pred    = model.predict(X)
    y_actual  = np.array(y_actual)
    deviation = (y_actual - y_pred) / (np.abs(y_pred) + 1e-9)

    if verbose:
        abs_dev = np.abs(deviation)
        n_total    = len(deviation)
        n_normal   = int((abs_dev  < DEVIATION_WARNING).sum())
        n_warning  = int(((abs_dev >= DEVIATION_WARNING) & (abs_dev < DEVIATION_CRITICAL)).sum())
        n_critical = int((abs_dev >= DEVIATION_CRITICAL).sum())

        print("  [괴리율 분포]")
        print(f"    정상   (|d| < {DEVIATION_WARNING:.0%}) : {n_normal:>5}건 ({n_normal/n_total:.1%})")
        print(f"    주의   ({DEVIATION_WARNING:.0%}~{DEVIATION_CRITICAL:.0%}) : {n_warning:>5}건 ({n_warning/n_total:.1%})")
        print(f"    고위험 (|d| ≥ {DEVIATION_CRITICAL:.0%}) : {n_critical:>5}건 ({n_critical/n_total:.1%})")
        print(f"    평균 |괴리율| : {abs_dev.mean():.3f}  최대: {abs_dev.max():.3f}")

    return deviation


def classify_deviation(deviation: np.ndarray) -> np.ndarray:
    """
    괴리율 배열 → 위험 레이블 배열 (0/1/2).

    Returns
    -------
    np.ndarray[int] — 0=정상, 1=주의, 2=고위험
    """
    abs_dev = np.abs(deviation)
    labels  = np.zeros(len(deviation), dtype=int)
    labels[abs_dev >= DEVIATION_WARNING]  = 1
    labels[abs_dev >= DEVIATION_CRITICAL] = 2
    return labels


# ──────────────────────────────────────────────────────────────
# 예측
# ──────────────────────────────────────────────────────────────

def predict_xgboost(model: XGBRegressor, X) -> np.ndarray:
    """학습된 XGBoost 모델로 가격 예측."""
    return model.predict(X)


# ──────────────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────────────

def _plot_learning_curve(model: XGBRegressor, fname: str = "xgb_learning_curve.png") -> None:
    """학습/검증 RMSE 커브 저장."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    results = model.evals_result()
    if not results:
        return

    train_rmse = results.get("validation_0", {}).get("rmse", [])
    val_rmse   = results.get("validation_1", {}).get("rmse", [])

    plt.figure(figsize=(9, 4))
    if train_rmse:
        plt.plot(train_rmse, label="Train RMSE", color="#4C72B0", linewidth=1.5)
    if val_rmse:
        plt.plot(val_rmse,   label="Val RMSE",   color="#C44E52", linewidth=1.5)
    if hasattr(model, "best_iteration"):
        plt.axvline(model.best_iteration, color="gray", linestyle="--",
                    label=f"Best iter ({model.best_iteration})")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title("XGBoost 학습 곡선")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, fname)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  학습 곡선 저장 → {save_path}")


def _plot_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label:  str = "XGBoost",
    fname:  str = "xgb_pred_vs_actual.png",
) -> None:
    """예측 vs 실제 산점도 저장."""
    os.makedirs(REPORT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.35, s=18, color="#4C72B0")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect fit")
    ax.set_xlabel("실제 거래금액 (만원)")
    ax.set_ylabel("예측 거래금액 (만원)")
    ax.set_title(f"{label} — 예측 vs 실제")
    ax.legend()

    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=40, color="#55A868", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="red", linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("잔차 (만원)")
    axes[1].set_ylabel("빈도")
    axes[1].set_title("잔차 분포")

    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, fname)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  예측 시각화 저장 → {save_path}")


def plot_feature_importance(
    model:         XGBRegressor,
    feature_names: List[str],
    top_n:         int = 20,
    fname:         str = "xgb_feature_importance.png",
) -> None:
    """XGBoost 내장 Feature Importance (gain 기준) 저장."""
    os.makedirs(REPORT_DIR, exist_ok=True)

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    names       = [feature_names[i] for i in indices]
    vals        = importances[indices]

    plt.figure(figsize=(10, max(4, top_n * 0.35)))
    plt.barh(names[::-1], vals[::-1], color="#4C72B0")
    plt.xlabel("Feature Importance (gain)")
    plt.title(f"XGBoost Feature Importance (Top {top_n})")
    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, fname)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Feature Importance 저장 → {save_path}")


# ──────────────────────────────────────────────────────────────
# 모델 저장 / 불러오기
# ──────────────────────────────────────────────────────────────

def save_xgboost(model: XGBRegressor, path: str = DEFAULT_MODEL_PATH) -> None:
    _ensure_dir(path)
    joblib.dump(model, path)
    size_kb = Path(path).stat().st_size / 1024
    print(f"  XGBoost 저장 완료 → {path}  ({size_kb:.1f} KB)")


def load_xgboost(path: str = DEFAULT_MODEL_PATH) -> XGBRegressor:
    if not os.path.exists(path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
    model = joblib.load(path)
    print(f"  XGBoost 불러오기 완료 ← {path}")
    return model


# ──────────────────────────────────────────────────────────────
# 단독 실행 테스트
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    더미 데이터로 전체 파이프라인 동작 확인.

    실행:
        python -m src.models.xgboost_regressor
    """
    from sklearn.model_selection import train_test_split

    print("\n" + "=" * 48)
    print("  XGBoost Regressor 단독 테스트 시작")
    print("=" * 48)

    rng  = np.random.default_rng(42)
    n    = 1200
    area = rng.uniform(40, 150, n)
    floor    = rng.integers(1, 30, n)
    year     = rng.integers(1990, 2024, n)
    hh       = rng.integers(50, 3000, n)
    dist     = rng.uniform(100, 5000, n)

    X = np.column_stack([area, floor, year, hh, dist])
    y = (area * 420 + floor * 250 - (2024 - year) * 60
         + hh * 0.6 - dist * 1.2 + rng.normal(0, 4000, n))

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    feature_names = ["전용면적", "층수", "건축연도", "단지세대수", "인근역거리"]

    model, metrics = train_xgboost(
        X_tr, y_tr, X_te, y_te,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        early_stopping=20,
        save_path="models/xgboost_regressor.pkl",
        plot=True,
        feature_names=feature_names,
    )

    # 괴리율 테스트
    deviation = compute_deviation_score(model, X_te, y_te, verbose=True)
    labels    = classify_deviation(deviation)

    print("\n  [괴리율 상위 5개]")
    top5 = np.argsort(np.abs(deviation))[::-1][:5]
    for rank, idx in enumerate(top5, 1):
        risk = ["🟢 정상", "🟡 주의", "🔴 고위험"][labels[idx]]
        print(f"    {rank}위 | 괴리율 {deviation[idx]:+.2%} | {risk}")

    print("\n  테스트 완료 ✓")
