"""
Linear Regression vs XGBoost 가격 예측 성능 비교 실행 스크립트

실행:
    python -m src.models.compare_models
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.linear_regression import train_linear, predict_linear
from src.models.xgboost_regressor import train_xgboost, predict_xgboost

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def main():
    print("\n================================================")
    print("  가격 예측 모델 학습 및 비교 (Linear vs XGBoost)")
    print("================================================")

    data_path = "data/processed/integrated_dataset_v1_with_s1.csv"
    if not os.path.exists(data_path):
        print(f"데이터 파일이 없습니다: {data_path}")
        return

    print("  데이터 로딩 중...")
    df = pd.read_csv(data_path, encoding="utf-8-sig", low_memory=False)

    features = ["전용면적(㎡)", "층", "건축년도"]
    target   = "거래금액(만원)"

    df = df.dropna(subset=features + [target])

    X = df[features].values
    y = df[target].values

    print(f"  총 {len(df):,}건의 데이터로 학습을 시작합니다.\n")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Linear Regression
    print("  [1] Linear Regression 모델 학습 중...")
    lr_model, lr_metrics = train_linear(
        X_tr, y_tr, X_te, y_te,
        cv_folds=5,
        save_path="models/linear_regression.pkl",
        plot=False,
    )
    lr_preds = predict_linear(lr_model, X_te)

    # 2. XGBoost
    print("\n  [2] XGBoost Regressor 모델 학습 중...")
    xgb_model, xgb_metrics = train_xgboost(
        X_tr, y_tr, X_te, y_te,
        feature_names=features,
        save_path="models/xgboost_regressor.pkl",
        plot=False,
    )
    xgb_preds = predict_xgboost(xgb_model, X_te)

    # 3. 비교 출력
    lr_r2, lr_rmse   = lr_metrics["R2"],  lr_metrics["RMSE"]
    xgb_r2, xgb_rmse = xgb_metrics["R2"], xgb_metrics["RMSE"]

    print("\n[가격 예측 모델 비교]")
    print(f"Linear Regression : R² = {lr_r2:.2f}, RMSE = {lr_rmse:,.0f}만원")
    print(f"XGBoost           : R² = {xgb_r2:.2f}, RMSE = {xgb_rmse:,.0f}만원")
    print("→ XGBoost 채택\n" if xgb_r2 > lr_r2 else "→ Linear Regression 채택\n")

    # 4. 비교 산점도
    os.makedirs("reports", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, preds, title, color, r2 in [
        (axes[0], lr_preds,  "Linear Regression", "#4C72B0", lr_r2),
        (axes[1], xgb_preds, "XGBoost Regressor", "#55A868", xgb_r2),
    ]:
        ax.scatter(y_te, preds, alpha=0.3, color=color, s=10)
        lim = [min(y_te.min(), preds.min()), max(y_te.max(), preds.max())]
        ax.plot(lim, lim, "r--", linewidth=1.2)
        ax.set_xlabel("실제가 (만원)")
        ax.set_ylabel("예측가 (만원)")
        ax.set_title(f"{title}\n(R²: {r2:.2f})")

    plt.tight_layout()
    plt.savefig("reports/model_comparison.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("비교 산점도 저장 완료 → reports/model_comparison.png")


if __name__ == "__main__":
    main()
