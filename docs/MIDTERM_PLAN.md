# 중간 발표 준비 작업 계획서

> 발표일: 2026. 5. 6. (화)
> 작성 기준일: 2026. 5. 4. (일)
> 작업 가능일: 5/4 (오늘), 5/5 (내일)

---

## 전체 의존성 흐름 (이 순서를 반드시 지켜야 합니다)

```
[문승준] CSV 다운로드 & 전처리
        ↓
        integrated_dataset_v1.csv 생성 및 공유
        ↓
        ┌─────────────────────────────┐
        ↓                             ↓
[손종인] peer_group.py          [안우현] xgboost_regressor.py
        K-Means 군집화                 Linear Regression 비교
        클러스터 시각화                 R², RMSE 결과표
        ↓                             ↓
        ┌─────────────────────────────┘
        ↓
[문승준] autoencoder.py
        재구성 오차 분포 그래프
        ↓
[안우현] finbert_sentiment.py (독립 작업, 언제든 가능)
        샘플 뉴스 감성 분석 결과
        ↓
[전체] 발표 자료 취합 (5/5 저녁)
```

---

## 핵심 규칙

```
문승준이 integrated_dataset_v1.csv 를 완성하기 전까지
손종인, 안우현은 모델 학습을 시작할 수 없습니다.

문승준의 최우선 목표 = 오늘(5/4) 오전 중 데이터셋 완성 후 공유
```

---

## 5/4 (오늘) 작업

---

### 문승준 — 데이터 파이프라인 완성 (최우선)

**목표: `data/processed/integrated_dataset_v1.csv` 생성 후 팀 공유**

#### Step 1. CSV 데이터 다운로드

아래 두 사이트에서 파일을 다운받아 `data/raw/` 폴더에 넣습니다.

**국토부 실거래가 (아파트 매매)**
1. [공공데이터포털](https://www.data.go.kr) 접속
2. "아파트매매 실거래 상세 자료" 검색
3. 2020년 ~ 2024년 연도별 CSV 다운로드 (5개 파일)
4. `data/raw/molit/` 폴더에 저장

**K-apt 관리비**
1. [K-apt](https://www.k-apt.go.kr) 접속 → 공개정보 → 관리비 공개
2. 원하는 지역·기간 선택 후 CSV 다운로드
3. `data/raw/kapt/` 폴더에 저장

> 파일이 크면 특정 지역(예: 경기도 성남시)만 먼저 받아도 됩니다.
> 중간 발표용이므로 전국 전체일 필요 없습니다.

---

#### Step 2. `src/utils/config.py` 작성

모든 파일에서 환경변수를 불러올 때 사용합니다.
가장 먼저 만들어야 합니다.

```python
# src/utils/config.py
from dotenv import load_dotenv
import os

load_dotenv()

MOLIT_API_KEY      = os.getenv("MOLIT_API_KEY")
KAPT_API_KEY       = os.getenv("KAPT_API_KEY")
NAVER_CLIENT_ID    = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET= os.getenv("NAVER_CLIENT_SECRET")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
```

---

#### Step 3. `src/data/batch_loader.py` 작성

`data/raw/` 폴더 안의 CSV 파일들을 전부 읽어서 하나로 합칩니다.

**이 파일이 만들어내는 것**: `molit_raw_df`, `kapt_raw_df` (pandas DataFrame)

```python
# src/data/batch_loader.py
import pandas as pd
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent.parent
RAW_DIR   = BASE_DIR / "data" / "raw"

def load_molit_csvs() -> pd.DataFrame:
    """data/raw/molit/ 안의 CSV 파일 전부 읽어서 합치기"""
    folder = RAW_DIR / "molit"
    files  = list(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"data/raw/molit/ 에 CSV 파일이 없습니다.")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="cp949")
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding="utf-8")
        dfs.append(df)
        print(f"  ✅ {f.name} 로드 완료 ({len(df):,}행)")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n국토부 전체: {len(combined):,}행")
    return combined


def load_kapt_csvs() -> pd.DataFrame:
    """data/raw/kapt/ 안의 CSV 파일 전부 읽어서 합치기"""
    folder = RAW_DIR / "kapt"
    files  = list(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"data/raw/kapt/ 에 CSV 파일이 없습니다.")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="cp949")
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding="utf-8")
        dfs.append(df)
        print(f"  ✅ {f.name} 로드 완료 ({len(df):,}행)")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nK-apt 전체: {len(combined):,}행")
    return combined


if __name__ == "__main__":
    molit_df = load_molit_csvs()
    kapt_df  = load_kapt_csvs()
    print("\n컬럼 확인 (국토부):", molit_df.columns.tolist())
    print("컬럼 확인 (K-apt):", kapt_df.columns.tolist())
```

> 실행해서 컬럼명을 확인한 뒤 아래 전처리에서 실제 컬럼명을 사용합니다.

---

#### Step 4. `src/features/preprocessor.py` 작성

**이 파일이 만들어내는 것**: 정제된 `molit_clean_df`, `kapt_clean_df`

```python
# src/features/preprocessor.py
import pandas as pd

def preprocess_molit(df: pd.DataFrame) -> pd.DataFrame:
    """
    국토부 실거래가 전처리
    - 컬럼명은 실제 CSV의 컬럼명으로 수정 필요
    """
    df = df.copy()

    # 1. 거래금액 숫자로 변환 (콤마 제거)
    df["거래금액"] = (
        df["거래금액"].astype(str)
        .str.replace(",", "")
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )

    # 2. 필수 컬럼 결측치 제거
    df = df.dropna(subset=["거래금액", "전용면적", "층"])

    # 3. 이상값 제거 (거래금액 0 이하, 전용면적 0 이하)
    df = df[df["거래금액"] > 0]
    df = df[df["전용면적"] > 0]

    # 4. 거래금액 단위 통일 (만원 → 억원)
    df["거래금액_억"] = df["거래금액"] / 10000

    # 5. 평당가 파생변수 생성
    df["평당가"] = df["거래금액"] / df["전용면적"]

    print(f"국토부 전처리 완료: {len(df):,}행")
    return df.reset_index(drop=True)


def preprocess_kapt(df: pd.DataFrame) -> pd.DataFrame:
    """
    K-apt 관리비 전처리
    - 컬럼명은 실제 CSV의 컬럼명으로 수정 필요
    """
    df = df.copy()

    # 필수 컬럼 결측치 제거 (단지코드, 전기사용량 등)
    essential = [c for c in ["단지코드", "전기사용량", "관리비"] if c in df.columns]
    df = df.dropna(subset=essential)

    print(f"K-apt 전처리 완료: {len(df):,}행")
    return df.reset_index(drop=True)
```

---

#### Step 5. `src/features/integrator.py` 작성

**이 파일이 만들어내는 것**: `data/processed/integrated_dataset_v1.csv` ← 팀 전체가 기다리는 파일

```python
# src/features/integrator.py
import pandas as pd
from pathlib import Path

BASE_DIR      = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def integrate(molit_df: pd.DataFrame, kapt_df: pd.DataFrame) -> pd.DataFrame:
    """
    단지코드 기준으로 실거래가 + 관리비 LEFT JOIN
    - 단지코드 컬럼명은 실제 데이터에 맞게 수정 필요
    """
    # 양쪽 데이터의 단지코드 컬럼명을 통일
    # 실제 컬럼명 확인 후 아래 수정
    molit_key = "단지코드"   # 국토부 CSV의 실제 컬럼명으로 변경
    kapt_key  = "단지코드"   # K-apt CSV의 실제 컬럼명으로 변경

    merged = pd.merge(
        molit_df, kapt_df,
        left_on=molit_key, right_on=kapt_key,
        how="left"
    )

    print(f"통합 완료: {len(merged):,}행, {len(merged.columns)}개 컬럼")
    return merged


def save_integrated(df: pd.DataFrame, version: str = "v1") -> Path:
    """통합 데이터셋 저장"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"integrated_dataset_{version}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"✅ 저장 완료: {path}")
    return path


if __name__ == "__main__":
    from src.data.batch_loader import load_molit_csvs, load_kapt_csvs
    from src.features.preprocessor import preprocess_molit, preprocess_kapt

    molit_raw  = load_molit_csvs()
    kapt_raw   = load_kapt_csvs()
    molit_clean = preprocess_molit(molit_raw)
    kapt_clean  = preprocess_kapt(kapt_raw)
    integrated  = integrate(molit_clean, kapt_clean)
    save_integrated(integrated)
```

> 완성 후 `integrated_dataset_v1.csv` 를 카카오톡 혹은 Google Drive로 팀원에게 공유합니다.
> (파일이 크면 data/raw 폴더에 함께 넣어주세요, gitignore 되어 있으므로 git에는 올라가지 않습니다.)

---

### 손종인 — API 수집 모듈 완성 (문승준과 병렬 진행)

**목표: 당월 데이터를 실제로 받아오는 모듈 3개 완성**

이 작업은 `integrated_dataset_v1.csv` 없이 독립적으로 진행합니다.

#### 해야 할 것
- `src/data/molit_api.py` : 국토부 당월 실거래가 API 호출 → JSON 반환 확인
- `src/data/kapt_api.py` : K-apt 당월 관리비 API 호출 → JSON 반환 확인
- `src/data/news_crawler.py` : 네이버 뉴스 검색 API로 키워드 뉴스 가져오기

#### 완료 기준
각 모듈을 실행했을 때 실제 데이터가 콘솔에 출력되면 완료입니다.

```bash
python -m src.data.molit_api   # 국토부 데이터 출력 확인
python -m src.data.kapt_api    # K-apt 데이터 출력 확인
python -m src.data.news_crawler  # 뉴스 데이터 출력 확인
```

---

### 안우현 — KR-FinBERT 감성 분석 (문승준과 병렬 진행)

**목표: 샘플 뉴스 텍스트 넣으면 감성 점수가 나오는 것 확인**

이 작업도 `integrated_dataset_v1.csv` 없이 독립적으로 진행합니다.

#### 해야 할 것
- `src/models/finbert_sentiment.py` 작성 및 실행 확인

#### 완료 기준

```python
# 이 코드를 실행했을 때 점수가 나오면 완료
texts = [
    "분당 재건축 호재 발표로 집값 급등 예상",
    "A단지 주민 소송으로 관리비 분쟁 심화",
    "금리 인상으로 부동산 거래 위축"
]
results = score_news(texts)
# 예상 출력: [0.12, 0.85, 0.71]  (0=긍정, 1=부정)
```

---

## 5/5 (내일) 작업

> 전제조건: 문승준이 오늘 `integrated_dataset_v1.csv` 공유 완료

---

### 문승준 — Autoencoder 학습

**입력**: `data/processed/integrated_dataset_v1.csv`
**출력**: 재구성 오차 분포 히스토그램 이미지 (`reports/autoencoder_error_dist.png`)

- `src/models/autoencoder.py` 작성 및 학습
- 정상 거래 / 이상 거래 오차 분포가 나뉘는 그래프 생성
- 이 그래프가 발표 핵심 시각화 자료입니다

---

### 손종인 — Peer Group 군집화

**입력**: `data/processed/integrated_dataset_v1.csv` (문승준에게 받기)
**출력**: 군집 시각화 이미지 (`reports/peer_group_clusters.png`)

- `src/models/peer_group.py` 작성
- K-Means 군집화 후 scatter plot 생성 (색깔로 클러스터 구분)
- Z-score 계산해서 이탈 거래 몇 건인지 숫자 확인

---

### 안우현 — 가격 예측 모델 학습 및 비교

**입력**: `data/processed/integrated_dataset_v1.csv` (문승준에게 받기)
**출력**: 모델 성능 비교표 + 예측가 vs 실제가 산점도 (`reports/model_comparison.png`)

- `src/models/linear_regression.py` 학습 → R², RMSE 기록
- `src/models/xgboost_regressor.py` 학습 → R², RMSE 기록
- 두 모델 성능 비교 후 XGBoost 채택 근거 만들기

**예상 결과 형태**:
```
[가격 예측 모델 비교]
Linear Regression : R² = 0.71, RMSE = 3,200만원
XGBoost           : R² = 0.89, RMSE = 1,800만원
→ XGBoost 채택
```

---

### 5/5 저녁 — 전체 취합 및 발표 자료 제작

| 항목 | 담당 | 내용 |
|------|------|------|
| EDA 시각화 | 문승준 | 거래금액 분포, 지역별 평당가, 결측치 현황 |
| 모델 결과 정리 | 안우현 | Linear vs XGBoost 비교표, 오차 그래프 |
| 파이프라인 시연 | 손종인 | API 수집 → 데이터 흐름 live 시연 |
| 발표 슬라이드 | 전체 | 각자 담당 부분 슬라이드 작성 후 합치기 |

---

## 중간 발표 슬라이드 구성 (참고)

```
1. 프로젝트 개요 (1장)
   - 문제 정의, Triple Check 개념

2. 시스템 아키텍처 (1장)
   - README의 전체 흐름도 사용

3. 데이터 파이프라인 (1~2장)
   - 수집 데이터 종류, 통합 데이터셋 컬럼 구성
   - EDA: 데이터 분포, 상관관계 히트맵

4. 모델 결과 (3장)
   - Linear vs XGBoost 성능 비교
   - Autoencoder 재구성 오차 분포
   - K-Means 군집 시각화

5. KR-FinBERT 감성 분석 결과 (1장)
   - 샘플 뉴스 → 감성 점수 출력 스크린샷

6. 남은 작업 계획 (1장)
   - Triple Check 통합, Logistic, SHAP, FastAPI

7. Q&A
```

---

## 체크리스트

### 문승준
- [ ] `data/raw/molit/` CSV 다운로드 완료
- [ ] `data/raw/kapt/` CSV 다운로드 완료
- [ ] `src/utils/config.py` 작성
- [ ] `src/data/batch_loader.py` 작성 및 실행 확인
- [ ] `src/features/preprocessor.py` 작성 및 실행 확인
- [ ] `src/features/integrator.py` 작성 및 실행 확인
- [ ] `data/processed/integrated_dataset_v1.csv` 팀원 공유 ← 오늘 최우선
- [ ] `src/models/autoencoder.py` 학습 완료 (내일)
- [ ] `reports/autoencoder_error_dist.png` 생성 (내일)

### 손종인
- [ ] `src/data/molit_api.py` 실제 데이터 출력 확인
- [ ] `src/data/kapt_api.py` 실제 데이터 출력 확인
- [ ] `src/data/news_crawler.py` 실제 데이터 출력 확인
- [ ] `integrated_dataset_v1.csv` 수령 (문승준에게)
- [ ] `src/models/peer_group.py` 학습 완료 (내일)
- [ ] `reports/peer_group_clusters.png` 생성 (내일)

### 안우현
- [ ] `src/models/finbert_sentiment.py` 샘플 결과 출력 확인
- [ ] `integrated_dataset_v1.csv` 수령 (문승준에게)
- [ ] `src/models/linear_regression.py` 학습 + 성능 기록 (내일)
- [ ] `src/models/xgboost_regressor.py` 학습 + 성능 기록 (내일)
- [ ] `reports/model_comparison.png` 생성 (내일)
