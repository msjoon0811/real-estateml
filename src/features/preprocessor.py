"""
국토부 / K-apt 데이터 전처리 모듈.
batch_loader.py 로 읽은 원본 DataFrame을 받아 정제된 DataFrame을 반환합니다.

⚠️ 컬럼명 주의: CSV를 열어서 실제 컬럼명을 확인한 뒤 MOLIT_COLS, KAPT_COLS 를 수정하세요.
"""
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── 실제 CSV 컬럼명으로 수정하세요 ────────────────────────────────────────────
MOLIT_COLS = {
    "거래금액":  "거래금액",   # 예) "물건금액(만원)" 등으로 돼 있을 수 있음
    "전용면적":  "전용면적",
    "층":       "층",
    "건축년도":  "건축년도",
    "법정동":    "법정동",
    "아파트":    "아파트",
    "단지코드":  "단지코드",   # 없으면 아파트명+주소 조합으로 대체
}

KAPT_COLS = {
    "단지코드":  "단지코드",
    "전기사용량": "전기사용량",
    "관리비":    "관리비합계",
}
# ─────────────────────────────────────────────────────────────────────────────


def preprocess_molit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    logger.info(f"국토부 전처리 시작: {len(df):,}행")

    # 거래금액 숫자 변환 (콤마·공백 제거)
    price_col = MOLIT_COLS["거래금액"]
    if price_col in df.columns:
        df[price_col] = (
            df[price_col].astype(str)
            .str.replace(",", "").str.strip()
            .pipe(pd.to_numeric, errors="coerce")
        )

    # 필수 컬럼 결측 제거
    required = [MOLIT_COLS["거래금액"], MOLIT_COLS["전용면적"], MOLIT_COLS["층"]]
    required = [c for c in required if c in df.columns]
    before = len(df)
    df = df.dropna(subset=required)
    logger.info(f"  결측 제거: {before - len(df):,}행 제거")

    # 이상값 제거
    df = df[df[price_col] > 0]
    df = df[df[MOLIT_COLS["전용면적"]] > 0]

    # 파생 변수
    df["거래금액_억"] = df[price_col] / 10000
    df["평당가"]     = df[price_col] / df[MOLIT_COLS["전용면적"]]

    logger.info(f"국토부 전처리 완료: {len(df):,}행")
    return df.reset_index(drop=True)


def preprocess_kapt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    logger.info(f"K-apt 전처리 시작: {len(df):,}행")

    required = [v for v in KAPT_COLS.values() if v in df.columns]
    before = len(df)
    df = df.dropna(subset=required)
    logger.info(f"  결측 제거: {before - len(df):,}행 제거")

    logger.info(f"K-apt 전처리 완료: {len(df):,}행")
    return df.reset_index(drop=True)
