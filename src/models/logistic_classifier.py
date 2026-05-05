"""
logistic_classifier.py
----------------------
Logistic Regression 기반 거래 위험도 3-class 분류기
  0 = 정상  |  1 = 주의  |  2 = 고위험

담당: 안우현

입력 피처 (S1~S4, 각 0~1 범위)
────────────────────────────────
  S1 : Autoencoder 재구성 오차 점수  (패턴 이상)
  S2 : XGBoost 가격 괴리율 점수      (가격 이상)
  S3 : K-Means Peer Group Z-score    (또래 비교 이상)
  S4 : KR-FinBERT 뉴스 감성 점수     (부정 확률)

사용 방법
─────────
from src.models.logistic_classifier import train_logistic, predict_risk

model, scaler = train_logistic(X_train, y_train, X_test, y_test)
labels, probas = predict_risk(model, scaler, X_new)
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize

# ──────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH  = "models/logistic_classifier.pkl"
DEFAULT_SCALER_PATH = "models/logistic_scaler.pkl"
REPORT_DIR          = "reports"

CLASS_NAMES = ["정상", "주의", "고위험"]
RISK_MAP: Dict[int, str] = {0: "🟢 정상", 1: "🟡 주의", 2: "🔴 고위험"}


# ──────────────────────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# 학습
# ──────────────────────────────────────────────────────────────

def train_logistic(
    X_train,
    y_train,
    X_test            = None,
    y_test            = None,
    max_iter:   int   = 1000,
    cv_folds:   int   = 5,
    C:          float = 1.0,
    save_path:  Optional[str] = None,
    scaler_path:Optional[str] = None,
    plot:       bool  = False,
) -> Tuple[LogisticRegression, StandardScaler]:
    """
    Logistic Regression 분류기 학습 · 평가 · (선택) 저장 · (선택) 시각화.

    Parameters
    ----------
    X_train     : array-like (n_samples, 4) — 학습 피처 [S1, S2, S3, S4]
    y_train     : array-like (n_samples,)   — 0/1/2 레이블
    X_test      : (선택) 검증 피처
    y_test      : (선택) 검증 레이블
    max_iter    : 최대 반복 횟수 (기본 1000)
    cv_folds    : Stratified K-Fold 교차 검증 횟수 (0이면 생략)
    C           : 정규화 강도 역수 (기본 1.0, 작을수록 강한 정규화)
    save_path   : 모델 저장 경로 (None이면 저장 안 함)
    scaler_path : 스케일러 저장 경로
    plot        : True면 혼동행렬 + ROC 곡선 저장

    Returns
    -------
    model  : 학습된 LogisticRegression
    scaler : 피팅된 StandardScaler
    """
    # ── 스케일링 ───────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # ── 모델 생성 및 학습 ──────────────────────────
    model = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",
        C=C,
        class_weight="balanced",   # 클래스 불균형 보정
        random_state=42,
    )
    model.fit(X_scaled, y_train)

    sep = "=" * 52
    print(sep)
    print("  [Logistic Classifier 학습 완료]")
    print(f"  Classes        : {model.classes_.tolist()}")
    print(f"  C (정규화)     : {C}")
    print(f"  n_iter_        : {model.n_iter_[0]}")

    # ── 교차 검증 ──────────────────────────────────
    if cv_folds > 1:
        skf   = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_f1 = cross_val_score(model, X_scaled, y_train,
                                cv=skf, scoring="f1_macro")
        print(f"  CV-F1({cv_folds}fold)   : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    # ── 검증 평가 ──────────────────────────────────
    if X_test is not None and y_test is not None:
        y_test_a      = np.array(y_test)
        X_test_scaled = scaler.transform(X_test)
        pred          = model.predict(X_test_scaled)
        probas        = model.predict_proba(X_test_scaled)

        print("\n  [분류 리포트 — 검증 세트]")
        print(classification_report(y_test_a, pred, target_names=CLASS_NAMES, digits=4))

        # AUC (OvR)
        y_bin = label_binarize(y_test_a, classes=[0, 1, 2])
        try:
            auc = roc_auc_score(y_bin, probas, multi_class="ovr", average="macro")
            print(f"  Macro AUC (OvR) : {auc:.4f}")
        except Exception:
            pass

        print("  [혼동 행렬]")
        cm = confusion_matrix(y_test_a, pred)
        print(cm)

        if plot:
            _plot_confusion_matrix(y_test_a, pred)
            _plot_roc_curves(y_test_a, probas)

    print(sep)

    # ── 저장 ───────────────────────────────────────
    if save_path or scaler_path:
        save_logistic(
            model,  scaler,
            model_path  = save_path  or DEFAULT_MODEL_PATH,
            scaler_path = scaler_path or DEFAULT_SCALER_PATH,
        )

    return model, scaler


# ──────────────────────────────────────────────────────────────
# 예측
# ──────────────────────────────────────────────────────────────

def predict_risk(
    model:  LogisticRegression,
    scaler: StandardScaler,
    X,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    위험도 레이블 및 확률 예측.

    Parameters
    ----------
    model  : 학습된 LogisticRegression
    scaler : fit된 StandardScaler
    X      : 원본(비스케일) 피처 행렬 (n_samples, 4)

    Returns
    -------
    labels : np.ndarray[int]   — 0/1/2 레이블
    probas : np.ndarray[float] — (n_samples, 3) 확률
    """
    X_scaled = scaler.transform(X)
    labels   = model.predict(X_scaled)
    probas   = model.predict_proba(X_scaled)
    return labels, probas


def predict_risk_single(
    model:  LogisticRegression,
    scaler: StandardScaler,
    s1: float, s2: float, s3: float, s4: float,
) -> Dict:
    """
    단일 샘플 위험도 예측 (상세 딕셔너리 반환).

    Returns
    -------
    dict
        {
          'label'      : int          (0/1/2),
          'risk'       : str          ('🟢 정상' 등),
          'confidence' : float        (최대 확률),
          'proba'      : dict         ({'정상': ..., '주의': ..., '고위험': ...})
        }
    """
    labels, probas = predict_risk(model, scaler, [[s1, s2, s3, s4]])
    label   = int(labels[0])
    proba_v = probas[0]
    return {
        "label":      label,
        "risk":       RISK_MAP[label],
        "confidence": float(max(proba_v)),
        "proba":      {cn: round(float(p), 4) for cn, p in zip(CLASS_NAMES, proba_v)},
    }


# ──────────────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────────────

def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fname:  str = "logistic_confusion_matrix.png",
) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred),
        display_labels=CLASS_NAMES,
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Logistic Classifier — 혼동 행렬")
    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, fname)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  혼동 행렬 저장 → {save_path}")


def _plot_roc_curves(
    y_true:  np.ndarray,
    probas:  np.ndarray,
    fname:   str = "logistic_roc_curves.png",
) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)

    y_bin    = label_binarize(y_true, classes=[0, 1, 2])
    colors   = ["#55A868", "#8172B2", "#C44E52"]

    plt.figure(figsize=(7, 5))
    for i, (cn, col) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probas[:, i])
        try:
            auc = roc_auc_score(y_bin[:, i], probas[:, i])
            plt.plot(fpr, tpr, color=col, linewidth=2, label=f"{cn} (AUC={auc:.3f})")
        except Exception:
            plt.plot(fpr, tpr, color=col, linewidth=2, label=cn)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Logistic Classifier — ROC 곡선 (OvR)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, fname)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  ROC 곡선 저장 → {save_path}")


def plot_decision_boundary_2d(
    model:  LogisticRegression,
    scaler: StandardScaler,
    X:      np.ndarray,
    y:      np.ndarray,
    feat_x: int = 0,
    feat_y: int = 2,
    feature_names: Optional[List[str]] = None,
    fname:  str  = "logistic_decision_boundary.png",
) -> None:
    """
    2개 피처 기준 결정 경계 시각화 (S1 vs S3 기본).

    Parameters
    ----------
    feat_x / feat_y : X축/Y축으로 사용할 피처 인덱스 (기본 0=S1, 2=S3)
    """
    os.makedirs(REPORT_DIR, exist_ok=True)

    names = feature_names or [f"S{i+1}" for i in range(X.shape[1])]
    X_sc  = scaler.transform(X)
    X2    = X_sc[:, [feat_x, feat_y]]

    h   = 0.02
    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 나머지 피처는 0으로 패딩
    n_feat   = X_sc.shape[1]
    grid_full = np.zeros((xx.ravel().shape[0], n_feat))
    grid_full[:, feat_x] = xx.ravel()
    grid_full[:, feat_y] = yy.ravel()

    Z = model.predict(grid_full).reshape(xx.shape)

    cmap_bg = plt.cm.get_cmap("Pastel1", 3)
    cmap_pt = plt.cm.get_cmap("Set1",    3)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_bg)
    for cls, cn in enumerate(CLASS_NAMES):
        mask = y == cls
        plt.scatter(X2[mask, 0], X2[mask, 1],
                    c=[cmap_pt(cls)], label=cn, edgecolors="k",
                    linewidths=0.4, s=30, alpha=0.85)
    plt.xlabel(f"{names[feat_x]} (스케일)")
    plt.ylabel(f"{names[feat_y]} (스케일)")
    plt.title(f"결정 경계 — {names[feat_x]} vs {names[feat_y]}")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, fname)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  결정 경계 저장 → {save_path}")


# ──────────────────────────────────────────────────────────────
# 모델 저장 / 불러오기
# ──────────────────────────────────────────────────────────────

def save_logistic(
    model:       LogisticRegression,
    scaler:      StandardScaler,
    model_path:  str = DEFAULT_MODEL_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH,
) -> None:
    _ensure_dir(model_path)
    _ensure_dir(scaler_path)
    joblib.dump(model,  model_path)
    joblib.dump(scaler, scaler_path)
    m_kb = Path(model_path).stat().st_size  / 1024
    s_kb = Path(scaler_path).stat().st_size / 1024
    print(f"  모델 저장  → {model_path}  ({m_kb:.1f} KB)")
    print(f"  스케일러   → {scaler_path}  ({s_kb:.1f} KB)")


def load_logistic(
    model_path:  str = DEFAULT_MODEL_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH,
) -> Tuple[LogisticRegression, StandardScaler]:
    for p in (model_path, scaler_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {p}")
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"  Logistic 모델 불러오기 완료 ← {model_path}")
    return model, scaler


# ──────────────────────────────────────────────────────────────
# 단독 실행 테스트
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    generate_synthetic.py 없이 내부 더미 데이터로 전체 흐름 확인.

    실행:
        python -m src.models.logistic_classifier
    """
    from sklearn.model_selection import train_test_split

    print("\n" + "=" * 52)
    print("  Logistic Classifier 단독 테스트 시작")
    print("=" * 52)

    rng = np.random.default_rng(42)
    n   = 1500

    # 정상 샘플 (1000)
    X_norm = rng.uniform(0.0, 0.35, (1000, 4))
    y_norm = np.zeros(1000, dtype=int)

    # 주의 샘플 (300)
    X_warn = rng.uniform(0.30, 0.65, (300, 4))
    y_warn = np.ones(300, dtype=int)

    # 고위험 샘플 (200)
    X_high = rng.uniform(0.60, 1.00, (200, 4))
    y_high = np.full(200, 2, dtype=int)

    X = np.vstack([X_norm, X_warn, X_high])
    y = np.concatenate([y_norm, y_warn, y_high])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    feature_names = ["S1_price", "S2_volume", "S3_sentiment", "S4_auxiliary"]

    model, scaler = train_logistic(
        X_tr, y_tr,
        X_test=X_te, y_test=y_te,
        max_iter=1000,
        cv_folds=5,
        save_path="models/logistic_classifier.pkl",
        scaler_path="models/logistic_scaler.pkl",
        plot=True,
    )

    # 결정 경계 시각화
    plot_decision_boundary_2d(model, scaler, X_te, y_te,
                              feat_x=0, feat_y=2,
                              feature_names=feature_names)

    # 단일 샘플 예측
    print("\n  [단일 샘플 예측]")
    for s1, s2, s3, s4, expected in [
        (0.1, 0.1, 0.1, 0.1, "정상"),
        (0.5, 0.4, 0.5, 0.3, "주의"),
        (0.9, 0.8, 0.9, 0.7, "고위험"),
    ]:
        result = predict_risk_single(model, scaler, s1, s2, s3, s4)
        match  = "✓" if expected in result["risk"] else "✗"
        print(f"  S=({s1},{s2},{s3},{s4}) → {result['risk']}  "
              f"신뢰도 {result['confidence']:.1%}  [{match} 예상: {expected}]")

    print("\n  테스트 완료 ✓  →  reports/ 폴더를 확인하세요.")
