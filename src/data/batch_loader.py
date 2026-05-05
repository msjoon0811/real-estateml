"""
과거 데이터 CSV/Excel 파일을 data/raw/ 에서 읽어 DataFrame으로 반환합니다.
- 국토부 실거래가: data/raw/molit/*.csv  (상단 15줄 안내문 skip)
- K-apt 관리비:   data/raw/kapt/*.xlsx  (상단 1줄 안내문 skip)
"""
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger   = get_logger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"


def load_molit_csvs() -> pd.DataFrame:
    """국토부 실거래가 CSV 전체 로드 (skiprows=15)"""
    folder = RAW_DIR / "molit"
    # 아파트식별정보.csv 는 컬럼 구조가 달라서 제외
    files  = sorted(f for f in folder.glob("*.csv") if "식별정보" not in f.name)
    if not files:
        raise FileNotFoundError(f"{folder} 에 CSV 파일이 없습니다.")

    logger.info("=== 국토부 CSV 로드 시작 ===")
    dfs = []
    for f in files:
        for enc in ("cp949", "utf-8-sig", "utf-8", "euc-kr"):
            try:
                df = pd.read_csv(f, encoding=enc, skiprows=15, low_memory=False)
                logger.info(f"  {f.name}: {len(df):,}행 ({enc})")
                dfs.append(df)
                break
            except (UnicodeDecodeError, Exception):
                continue

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"  합계: {len(combined):,}행\n")
    return combined


def load_kapt_excels() -> pd.DataFrame:
    """K-apt 관리비 Excel 전체 로드 (header=1, 0번째 줄 안내문 skip)"""
    folder = RAW_DIR / "kapt"
    files  = sorted(list(folder.glob("*.xlsx")) + list(folder.glob("*.xls")))
    if not files:
        raise FileNotFoundError(f"{folder} 에 엑셀 파일이 없습니다.")

    logger.info("=== K-apt Excel 로드 시작 ===")
    dfs = []
    for f in files:
        df = pd.read_excel(f, engine="openpyxl", header=1)
        logger.info(f"  {f.name}: {len(df):,}행")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"  합계: {len(combined):,}행\n")
    return combined


if __name__ == "__main__":
    molit_df = load_molit_csvs()
    kapt_df  = load_kapt_excels()
    print("\n[국토부 컬럼]", molit_df.columns.tolist())
    print("\n[K-apt 컬럼]", kapt_df.columns.tolist())
    print(f"\n국토부: {len(molit_df):,}행 / K-apt: {len(kapt_df):,}행")
