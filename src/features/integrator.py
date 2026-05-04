"""
전처리된 국토부 + K-apt 데이터를 단지코드 기준으로 병합하고
data/processed/integrated_dataset_v1.csv 로 저장합니다.
"""
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger        = get_logger(__name__)
BASE_DIR      = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# ── 실제 CSV의 단지코드 컬럼명으로 수정하세요 ──────────────────────────────────
MOLIT_KEY = "단지코드"
KAPT_KEY  = "단지코드"
# ─────────────────────────────────────────────────────────────────────────────


def integrate(molit_df: pd.DataFrame, kapt_df: pd.DataFrame) -> pd.DataFrame:
    """국토부 + K-apt LEFT JOIN (단지코드 기준)"""
    logger.info("데이터 통합 시작...")

    if MOLIT_KEY not in molit_df.columns:
        logger.warning(f"국토부에 '{MOLIT_KEY}' 컬럼 없음 — preprocessor.py의 MOLIT_COLS 확인 필요")
    if KAPT_KEY not in kapt_df.columns:
        logger.warning(f"K-apt에 '{KAPT_KEY}' 컬럼 없음 — preprocessor.py의 KAPT_COLS 확인 필요")

    merged = pd.merge(
        molit_df, kapt_df,
        left_on=MOLIT_KEY, right_on=KAPT_KEY,
        how="left", suffixes=("", "_kapt")
    )

    # K-apt 매칭률 확인
    matched = merged[KAPT_KEY].notna().sum()
    rate    = matched / len(merged) * 100
    logger.info(f"K-apt 매칭: {matched:,}건 / {len(merged):,}건 ({rate:.1f}%)")
    logger.info(f"통합 완료: {len(merged):,}행, {len(merged.columns)}개 컬럼")
    return merged


def save_integrated(df: pd.DataFrame, version: str = "v1") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"integrated_dataset_{version}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"✅ 저장 완료: {path}")
    return path


if __name__ == "__main__":
    from src.data.batch_loader import load_molit_csvs, load_kapt_csvs
    from src.features.preprocessor import preprocess_molit, preprocess_kapt

    molit_raw   = load_molit_csvs()
    kapt_raw    = load_kapt_csvs()
    molit_clean = preprocess_molit(molit_raw)
    kapt_clean  = preprocess_kapt(kapt_raw)
    integrated  = integrate(molit_clean, kapt_clean)
    save_integrated(integrated)
