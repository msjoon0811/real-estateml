"""
아파트 실거래가 + 식별정보 통합 매핑 스크립트
================================================
사용법:
  1. DATA_DIR에 실거래가XXYY.csv 파일들을 넣으세요 (10개)
  2. ID_FILE 경로에 아파트식별정보.csv를 넣으세요
  3. python merge_apt_data.py 실행

매핑 방식:
  1차) 주소키 (시군구+본번+부번) → 정확 매핑
  2차) 단지명+건축년도           → 보완 매핑
"""

import pandas as pd
import glob
import re
import os
import unicodedata

# ────────────────────────────────────────────
# ★ 경로 설정 (여기만 수정하세요)
# ────────────────────────────────────────────
DATA_DIR = "./"                      # 실거래가 CSV들이 있는 폴더
ID_FILE  = "./아파트식별정보.csv"     # 식별정보 파일
OUT_FILE = "./merged_apt_final.csv"  # 출력 파일

# ────────────────────────────────────────────
# 내부 함수
# ────────────────────────────────────────────
def nfc(s):
    """macOS NFD 파일명 → NFC 변환 (한글 파일명 인식용)"""
    return unicodedata.normalize("NFC", s)

def extract_addr_key(addr):
    """'서울특별시 종로구 창신동 232' → '서울특별시 종로구 창신동 0232-0000'"""
    addr = str(addr).strip()
    m = re.search(r"(\d+)(?:-(\d+))?\s*$", addr)
    if m:
        bon  = m.group(1).zfill(4)
        bu   = (m.group(2) or "0").zfill(4)
        base = re.sub(r"\s*\d+(?:-\d+)?\s*$", "", addr).strip()
        return f"{base} {bon}-{bu}"
    return None

# ────────────────────────────────────────────
# 1. 실거래가 파일 전체 로드
# ────────────────────────────────────────────
print("=" * 55)
print("1단계: 실거래가 파일 로드 중...")

all_csv     = glob.glob(os.path.join(DATA_DIR, "*.csv"))
trade_files = [f for f in all_csv if "실거래가" in nfc(os.path.basename(f))]

if not trade_files:
    raise FileNotFoundError(
        f"실거래가 파일을 찾을 수 없어요.\n"
        f"경로: {DATA_DIR}\n"
        f"파일명이 '실거래가'로 시작하는지 확인하세요."
    )

print(f"   발견된 파일 {len(trade_files)}개:")
df_list = []
for f in sorted(trade_files):
    df = pd.read_csv(f, encoding="euc-kr", skiprows=15, dtype=str)
    df["_출처파일"] = os.path.basename(f)
    df_list.append(df)
    print(f"   ✅ {os.path.basename(f)}: {len(df):,}건")

trade = pd.concat(df_list, ignore_index=True)
print(f"\n   ▶ 합계: {len(trade):,}건")

# ────────────────────────────────────────────
# 2. 실거래가 전처리
# ────────────────────────────────────────────
print("\n2단계: 실거래가 전처리 중...")

trade.columns = trade.columns.str.strip()

# 헤더 잔재 행 제거
trade = trade[trade["NO"].str.strip().str.match(r"^\d+$", na=False)].copy()

# 타입 변환
trade["본번"]        = pd.to_numeric(trade["본번"], errors="coerce").fillna(0).astype(int)
trade["부번"]        = pd.to_numeric(trade["부번"], errors="coerce").fillna(0).astype(int)
trade["건축년도"]    = pd.to_numeric(trade["건축년도"], errors="coerce")
trade["거래금액(만원)"] = pd.to_numeric(
    trade["거래금액(만원)"].str.replace(",", ""), errors="coerce"
)

# 해제 건 제거
before = len(trade)
trade = trade[
    trade["해제사유발생일"].str.strip().isin(["-", "", "nan"]) |
    trade["해제사유발생일"].isna()
].copy()
print(f"   해제 건 제거: {before - len(trade):,}건 → {len(trade):,}건 남음")

# 주소 매핑 키 생성
trade["_주소키"] = (
    trade["시군구"].str.strip() + " " +
    trade["본번"].astype(str).str.zfill(4) + "-" +
    trade["부번"].astype(str).str.zfill(4)
)

# ────────────────────────────────────────────
# 3. 식별정보 로드
# ────────────────────────────────────────────
print("\n3단계: 식별정보 로드 중...")

id_df = pd.read_csv(ID_FILE, encoding="utf-8", dtype=str)
id_df.columns = id_df.columns.str.strip()

# 아파트(단지종류=1)만 필터
id_apt = id_df[id_df["단지종류"].str.strip() == "1"].copy()
print(f"   아파트 단지: {len(id_apt):,}개 (전체 {len(id_df):,}개 중)")

# 주소키 & 승인연도 생성
id_apt["_주소키"]   = id_apt["주소"].apply(extract_addr_key)
id_apt["_승인연도"] = pd.to_numeric(id_apt["사용승인일"].str[:4], errors="coerce")

# ────────────────────────────────────────────
# 4. 1차 매핑: 주소키 기준
# ────────────────────────────────────────────
print("\n4단계: 1차 매핑 (시군구+본번+부번) 중...")

ID_COLS = ["단지고유번호", "필지고유번호", "단지명_공시가격",
           "단지명_건축물대장", "단지명_도로명주소",
           "동수", "세대수", "사용승인일", "_승인연도"]

id_slim = id_apt[ID_COLS + ["_주소키"]].drop_duplicates(subset=["_주소키"])
merged  = pd.merge(trade, id_slim, on="_주소키", how="left")

m1 = merged["단지고유번호"].notna().sum()
print(f"   1차 성공: {m1:,} / {len(merged):,} ({m1/len(merged)*100:.1f}%)")

# ────────────────────────────────────────────
# 5. 2차 매핑: 단지명 + 건축년도 보완
# ────────────────────────────────────────────
print("\n5단계: 2차 매핑 (단지명+건축년도) 중...")

id_apt["_이름연도키"] = (
    id_apt["단지명_공시가격"].str.strip() + "_" +
    id_apt["_승인연도"].astype(str)
)
id_name = id_apt.drop_duplicates(subset=["_이름연도키"])[ID_COLS + ["_이름연도키"]]

unmatch = merged[merged["단지고유번호"].isna()].copy()
unmatch["_이름연도키"] = (
    unmatch["단지명"].str.strip() + "_" +
    unmatch["건축년도"].astype(str)
)
# 1차에서 붙은 식별 컬럼 제거 후 재매핑
unmatch2 = unmatch.drop(columns=[c for c in ID_COLS if c in unmatch.columns])
comp = pd.merge(unmatch2, id_name, on="_이름연도키", how="left")

m2 = comp["단지고유번호"].notna().sum()
print(f"   2차 성공: {m2:,}건 추가")

# ────────────────────────────────────────────
# 6. 최종 합치기 & 저장
# ────────────────────────────────────────────
print("\n6단계: 최종 합치기 & 저장 중...")

final = pd.concat([
    merged[merged["단지고유번호"].notna()],
    comp
], ignore_index=True)

# 내부 키 컬럼 제거
final.drop(columns=["_주소키", "_이름연도키"], errors="ignore", inplace=True)

total = final["단지고유번호"].notna().sum()

print(f"\n{'='*55}")
print(f"  전체 거래 건수   : {len(final):,}건")
print(f"  식별정보 매핑 성공: {total:,}건 ({total/len(final)*100:.1f}%)")
print(f"  미매핑 건수       : {len(final)-total:,}건")
print(f"{'='*55}")

os.makedirs(os.path.dirname(OUT_FILE) or ".", exist_ok=True)
final.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
print(f"\n✅ 저장 완료: {OUT_FILE}")
print(f"   최종 컬럼: {final.columns.tolist()}")
