"""
과거 5년치 CSV 파일을 data/raw/ 에서 읽어 DataFrame으로 반환합니다.
다운받은 CSV를 data/raw/molit/, data/raw/kapt/ 에 넣고 실행하세요.
"""
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger  = get_logger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"


def _read_csv_folder(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"{folder} 안에 CSV 파일이 없습니다. 파일을 넣고 다시 실행하세요.")

    dfs = []
    for f in files:
        for enc in ("cp949", "utf-8", "utf-8-sig"):
            try:
                df = pd.read_csv(f, encoding=enc, low_memory=False)
                logger.info(f"  로드: {f.name}  ({len(df):,}행)")
                dfs.append(df)
                break
            except (UnicodeDecodeError, Exception):
                continue

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"  합계: {len(combined):,}행\n")
    return combined


def load_molit_csvs() -> pd.DataFrame:
    """국토부 실거래가 CSV 전체 로드"""
    logger.info("=== 국토부 CSV 로드 시작 ===")
    return _read_csv_folder(RAW_DIR / "molit")


def load_kapt_csvs() -> pd.DataFrame:
    """K-apt 관리비 CSV 전체 로드"""
    logger.info("=== K-apt CSV 로드 시작 ===")
    return _read_csv_folder(RAW_DIR / "kapt")


if __name__ == "__main__":
    molit_df = load_molit_csvs()
    kapt_df  = load_kapt_csvs()

    print("\n[국토부 컬럼 목록]")
    print(molit_df.columns.tolist())
    print("\n[K-apt 컬럼 목록]")
    print(kapt_df.columns.tolist())
