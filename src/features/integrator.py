"""
전처리된 국토부 + K-apt 데이터를 (시군구 + 단지명) 기준으로 병합합니다.
두 데이터의 단지코드 체계가 달라 단지명+시군구를 공통 키로 사용합니다.
"""
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger        = get_logger(__name__)
BASE_DIR      = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def _normalize_name(series: pd.Series) -> pd.Series:
    """단지명 표기 차이 최소화 (공백·특수문자 제거, 소문자 통일)"""
    return series.astype(str).str.replace(r"[\s\-_]", "", regex=True).str.lower()


def integrate(molit_df: pd.DataFrame, kapt_df: pd.DataFrame) -> pd.DataFrame:
    """
    국토부 + K-apt 병합
    - 키: (시군구, 단지명) — 정규화 후 매칭
    - 국토부 기준 LEFT JOIN (거래 데이터 중심)
    """
    logger.info("데이터 통합 시작...")

    molit = molit_df.copy()
    kapt  = kapt_df.copy()

    # 매칭 키 정규화
    # 국토부 시군구: "서울특별시 노원구 상계동" → "서울특별시노원구" (앞 두 단어)
    molit["_key_gu"] = molit["시군구"].astype(str).str.split().str[:2].str.join("").str.lower()
    molit["_key_name"] = _normalize_name(molit["단지명"])

    # K-apt 시군구: 시도("서울특별시") + 시군구("노원구") → "서울특별시노원구"
    kapt["_key_gu"] = (kapt["시도"].astype(str) + kapt["시군구"].astype(str)).str.lower()
    kapt["_key_name"] = _normalize_name(kapt["단지명"])

    # K-apt: 단지별 최신 1건만 사용 (중복 방지)
    kapt_dedup = (
        kapt.sort_values("발생년월(YYYYMM)", ascending=False)
        .drop_duplicates(subset=["_key_name", "_key_gu"])
    )

    merged = pd.merge(
        molit, kapt_dedup,
        on=["_key_name", "_key_gu"],
        how="left",
        suffixes=("", "_kapt")
    )

    # 임시 키 컬럼 제거
    merged = merged.drop(columns=["_key_name", "_key_gu"])

    # 매칭률 출력
    matched = merged["단지코드"].notna().sum()
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
    from src.data.batch_loader import load_molit_csvs, load_kapt_excels
    from src.features.preprocessor import preprocess_molit, preprocess_kapt

    molit_raw   = load_molit_csvs()
    kapt_raw    = load_kapt_excels()
    molit_clean = preprocess_molit(molit_raw)
    kapt_clean  = preprocess_kapt(kapt_raw)
    integrated  = integrate(molit_clean, kapt_clean)
    save_integrated(integrated)
