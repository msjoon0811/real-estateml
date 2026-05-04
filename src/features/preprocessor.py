"""
국토부 / K-apt 데이터 전처리 모듈.
실제 컬럼명 기준으로 정제하고 파생 변수를 생성합니다.
"""
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def preprocess_molit(df: pd.DataFrame) -> pd.DataFrame:
    """
    국토부 실거래가 전처리
    주요 컬럼: 시군구, 단지명, 전용면적(㎡), 계약년월, 거래금액(만원), 층, 건축년도
    """
    df = df.copy()
    logger.info(f"국토부 전처리 시작: {len(df):,}행")

    # 1. 거래금액 숫자 변환 (콤마 제거)
    df["거래금액(만원)"] = (
        df["거래금액(만원)"].astype(str)
        .str.replace(",", "").str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )

    # 2. 전용면적 숫자 변환
    df["전용면적(㎡)"] = pd.to_numeric(df["전용면적(㎡)"], errors="coerce")

    # 3. 층 숫자 변환
    df["층"] = pd.to_numeric(df["층"], errors="coerce")

    # 4. 건축년도 숫자 변환
    df["건축년도"] = pd.to_numeric(df["건축년도"], errors="coerce")

    # 5. 계약년월 문자열로 통일 (예: 202205)
    df["계약년월"] = df["계약년월"].astype(str).str.strip()

    # 6. 필수 컬럼 결측 제거
    before = len(df)
    df = df.dropna(subset=["거래금액(만원)", "전용면적(㎡)", "층"])
    logger.info(f"  결측 제거: {before - len(df):,}행 제거")

    # 7. 이상값 제거
    df = df[(df["거래금액(만원)"] > 0) & (df["전용면적(㎡)"] > 0)]

    # 8. 파생 변수
    df["거래금액_억"]  = df["거래금액(만원)"] / 10000
    df["평당가(만원)"] = df["거래금액(만원)"] / df["전용면적(㎡)"]

    # 9. 필요한 컬럼만 선택
    keep = ["시군구", "단지명", "전용면적(㎡)", "계약년월", "계약일",
            "거래금액(만원)", "거래금액_억", "평당가(만원)", "층", "건축년도", "거래유형"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]

    logger.info(f"국토부 전처리 완료: {len(df):,}행")
    return df.reset_index(drop=True)


def preprocess_kapt(df: pd.DataFrame) -> pd.DataFrame:
    """
    K-apt 관리비 전처리
    주요 컬럼: 단지코드, 단지명, 발생년월, 공용관리비계, 전기료(전용), 장충금 총적립금액
    """
    df = df.copy()
    logger.info(f"K-apt 전처리 시작: {len(df):,}행")

    # 1. 단지코드 결측 제거
    before = len(df)
    df = df.dropna(subset=["단지코드"])
    logger.info(f"  단지코드 결측 제거: {before - len(df):,}행 제거")

    # 2. 발생년월 문자열 통일
    df["발생년월(YYYYMM)"] = df["발생년월(YYYYMM)"].astype(str).str.strip()

    # 3. 수치 컬럼 변환
    numeric_cols = ["공용관리비계", "전기료(전용)", "전기료(공용)", "장충금 총적립금액",
                    "난방비(전용)", "수도료(전용)"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. 필요한 컬럼만 선택
    keep = ["시도", "시군구", "단지코드", "단지명", "발생년월(YYYYMM)",
            "공용관리비계", "전기료(전용)", "전기료(공용)",
            "난방비(전용)", "수도료(전용)", "장충금 총적립금액"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]

    logger.info(f"K-apt 전처리 완료: {len(df):,}행")
    return df.reset_index(drop=True)
