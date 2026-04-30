import pandas as pd
import os
from pathlib import Path

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def load_data():
    """
    data/raw 폴더에 있는 국토부 실거래가 CSV와 K-apt 관리비 CSV를 불러옵니다.
    (파일명은 실제 다운받은 파일명으로 추후 수정하세요)
    """
    print("데이터 로딩 중...")
    try:
        # ⚠️ 실제 파일명으로 수정 필요!
        molit_df = pd.read_csv(RAW_DIR / "molit_transaction_sample.csv", encoding="utf-8")
        kapt_df = pd.read_csv(RAW_DIR / "kapt_management_sample.csv", encoding="utf-8")
        return molit_df, kapt_df
    except FileNotFoundError as e:
        print(f"❌ 에러: 파일을 찾을 수 없습니다. data/raw 폴더에 CSV 파일을 넣어주세요.\n{e}")
        return None, None

def preprocess_molit(df):
    """국토부 실거래가 데이터 전처리 (결측치 제거, 날짜 타입 변환 등)"""
    print("실거래가 데이터 전처리 중...")
    # 예시: '거래금액' 컬럼의 콤마(,) 제거 후 숫자로 변환
    if '거래금액' in df.columns:
        df['거래금액'] = df['거래금액'].astype(str).str.replace(',', '').astype(int)
    return df

def preprocess_kapt(df):
    """K-apt 관리비 및 에너지 데이터 전처리"""
    print("K-apt 데이터 전처리 중...")
    return df

def merge_datasets(molit_df, kapt_df):
    """
    두 데이터프레임을 특정 기준(예: 아파트 단지 코드, 거래 연월)으로 병합(Merge)합니다.
    """
    print("데이터 병합(Merge) 중...")
    # ⚠️ 병합 기준이 되는 Key 컬럼명은 실제 데이터에 맞게 수정해야 합니다.
    # 예: merged_df = pd.merge(molit_df, kapt_df, left_on='단지명', right_on='단지명', how='left')
    
    # 지금은 가봉합 상태이므로 임시로 반환
    merged_df = molit_df # 임시
    return merged_df

def feature_engineering(df):
    """Autoencoder 학습에 사용할 파생 변수(Feature) 생성"""
    print("파생 변수(Feature Engineering) 생성 중...")
    # 예: df['평당단가'] = df['거래금액'] / df['전용면적']
    return df

def main():
    # 1. 데이터 로드
    molit_df, kapt_df = load_data()
    
    if molit_df is None or kapt_df is None:
        return

    # 2. 개별 전처리
    molit_df = preprocess_molit(molit_df)
    kapt_df = preprocess_kapt(kapt_df)

    # 3. 데이터 병합
    merged_df = merge_datasets(molit_df, kapt_df)

    # 4. 파생 변수 생성
    final_df = feature_engineering(merged_df)

    # 5. 결과 저장
    output_path = PROCESSED_DIR / "integrated_dataset_v1.csv"
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 통합 데이터셋 저장 완료! 경로: {output_path}")

if __name__ == "__main__":
    main()
