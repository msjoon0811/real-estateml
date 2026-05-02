# 머신러닝 앙상블 기반 부동산 이상 거래 탐지 및 저평가 매물 식별 시스템

> SW중심대학 산학협력 프로젝트 (2026학년도 1학기 기업연계 프로젝트)
> 선문대학교 컴퓨터공학부 · 기계학습프로젝트

실거래가(수치) · 관리비(물리) · 뉴스(텍스트) 세 가지 이종 데이터를 교차 검증해
부동산 시장의 자전거래 · 시세 조작 · 저평가 매물을 탐지하고,
그 근거를 설명 가능한 형태(XAI)로 제공하는 머신러닝 파이프라인 시스템입니다.

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [전체 아키텍처](#2-전체-아키텍처)
3. [모델 설명 (6개)](#3-모델-설명)
4. [기술 스택](#4-기술-스택)
5. [디렉토리 구조](#5-디렉토리-구조)
6. [역할 분담 및 작업 가이드](#6-역할-분담-및-작업-가이드)
7. [로컬 환경 설정 (처음 시작하는 법)](#7-로컬-환경-설정)
8. [일정 (마일스톤)](#8-일정-마일스톤)
9. [보안 주의사항](#9-보안-주의사항)
10. [팀 연락망](#10-팀-연락망)

---

## 1. 프로젝트 개요

### 문제 정의

기존 부동산 플랫폼(네이버부동산 · 직방 등)은 신고된 실거래가를 수동으로 나열할 뿐,
해당 거래가 실제 거래인지(자전거래 여부), 적정 가격 대비 얼마나 저평가 · 고평가됐는지를
검증하는 수단이 없습니다. 일반 투자자와 실거주자는 거래의 진위와 객관적 가치를
판단하기 매우 어렵습니다.

### 해결 접근 — Triple Check

세 가지 이종 데이터를 독립적으로 분석한 뒤 교차 검증합니다.

| 체크 | 데이터 | 모델 | 탐지 내용 |
|------|--------|------|-----------|
| **수치 체크** | 실거래가 | XGBoost + Autoencoder | 가격 괴리 & 패턴 이상 |
| **물리 체크** | 관리비 · 에너지 사용량 | Peer Group (K-Means) | 실거주 여부 확인 |
| **상황 체크** | 뉴스 텍스트 | KR-FinBERT | 호재/악재 실제 존재 여부 |

세 체크의 점수를 **Logistic Regression**이 종합해 최종 위험도를 출력하고,
**SHAP**이 "왜 이 거래가 의심스러운가"를 변수별로 설명합니다.

---

## 2. 전체 아키텍처

### 두 개의 시간축 이해하기

```
[오프라인 - 프로젝트 초반 한 번만]
  과거 5년치 데이터로 모델 6개를 학습하고 저장

[온라인 - 매일 자동 실행]
  당월 신규 거래 데이터를 수집 → 학습된 모델로 위험도 판정 → 결과 출력
```

---

### 전체 흐름도

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 1. 데이터 수집 (Two-Track)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [학습용 - 과거 5년치 CSV 다운로드]     [추론용 - 당월 API 실시간 수집]
  ┌───────────────────────────┐         ┌──────────────────────────┐
  │ 국토부 실거래가 CSV         │         │ 국토부 실거래가 (이번 달) │
  │ K-apt 관리비 CSV           │    +    │ K-apt 관리비 (이번 달)   │
  │ (공공데이터포털에서 수동DL) │         │ 네이버 뉴스 크롤링        │
  └───────────────────────────┘         └──────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 2. 전처리 & 통합
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  단지 코드를 기준으로 세 데이터를 하나의 테이블로 합칩니다.

  단지코드 | 거래가 | 면적 | 층 | 준공연도 | 관리비 | 전기세 | 뉴스점수
  A001    | 8억   | 84㎡ | 10 | 2015   | 25만  | 150kWh | -0.3

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 3. 오프라인 학습 (한 번 실행)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  과거 5년 데이터 →
  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
  │ Linear          │  │    XGBoost      │  │  Autoencoder    │
  │ Regression      │  │   (가격 예측)    │  │ (정상 패턴 학습) │
  │ (베이스라인)     │  │                 │  │                 │
  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
           │                    │                     │
           └──────────┬─────────┘                     │
                      ▼                               │
             성능 비교 (R², RMSE)                      │
             → XGBoost가 우수 → 채택 확정              │
                                                      │
  ┌─────────────────┐  ┌───────────────────────────────┘
  │     K-Means     │  │  인공 이상치 데이터 생성
  │ (Peer Group 구성)│  │  (정상 데이터 가격 ±40% 조작)
  │ 단지별 cluster  │  │         ↓
  │ id 부여         │  │  Logistic Regression 학습
  └─────────────────┘  │  [S1,S2,S3,S4] → 위험도 분류
                       └──────────────────────────────

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 4. 온라인 추론 (매일 자동)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [신규 거래 1건 입력]
  예) 분당구 A단지 84㎡ 10층 → 8억 거래
                 │
    ┌────────────┼────────────┬────────────────┐
    ▼            ▼            ▼                ▼
 ┌────────┐ ┌────────┐ ┌──────────┐ ┌──────────────┐
 │XGBoost │ │AutoEnc-│ │  Peer    │ │  KR-FinBERT  │
 │        │ │oder    │ │  Group   │ │              │
 │적정가  │ │        │ │          │ │  뉴스 감성    │
 │9억 예측│ │패턴 이상│ │또래 비교 │ │  분석        │
 │괴리율  │ │오차 계산│ │Z-score  │ │              │
 │-11%   │ │높음    │ │이탈      │ │  부정: 0.3   │
 └───┬────┘ └───┬────┘ └────┬─────┘ └──────┬───────┘
     │          │           │              │
     ▼          ▼           ▼              ▼
   S2: 0.6   S1: 0.8    S3: 0.5        S4: 0.3
     │          │           │              │
     └──────────┴───────────┴──────────────┘
                            │
                            ▼
               ┌─────────────────────────┐
               │    Logistic Regression  │
               │      최종 위험도 분류    │
               └────────────┬────────────┘
                            │
                            ▼
                       🟡 주의 등급
                            │
                            ▼
               ┌─────────────────────────┐
               │          SHAP           │
               │  "S1이 가장 큰 원인"     │
               │  변수별 기여도 시각화    │
               └─────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 5. 출력
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  🔴 고위험 (score >= 0.7) → 텔레그램 즉시 알림
  🟡 주의   (score >= 0.4) → 웹 대시보드 등록
  🟢 정상   (score  < 0.4) → 기록만 저장
```

---

## 3. 모델 설명

> 각 모델이 왜 필요한지, 무엇을 하는지 정리합니다.

### 모델 1 — Linear Regression (다중 선형 회귀)

- **역할**: XGBoost의 베이스라인 비교용
- **입력**: 전용면적, 층, 준공연도, 세대수, 시군구 코드 등
- **출력**: 예측 아파트 가격
- **왜 필요한가**: "XGBoost가 왜 더 나은지"를 숫자로 증명하기 위해 비교군으로 사용합니다.
  R² 와 RMSE 지표로 XGBoost와 비교해 XGBoost 채택 근거를 만듭니다.

### 모델 2 — XGBoost Regressor (그래디언트 부스팅 회귀)

- **역할**: 적정 거래가 예측 → 저평가 / 고가 거래 탐지
- **입력**: Linear Regression과 동일한 피처
- **출력**: 예측 적정가, 괴리율 = (실제가 - 예측가) / 예측가
- **왜 필요한가**: 괴리율이 크게 음수이면 저평가 매물, 크게 양수이면 고가 신고 의심입니다.
  트리 기반이라 비선형 관계도 잘 잡고, SHAP과 궁합이 좋습니다.

### 모델 3 — Autoencoder (오토인코더)

- **역할**: 거래 전체 패턴의 이상 탐지
- **입력**: 거래가, 관리비, 전기세, 층, 면적 등 수치 피처 전체
- **출력**: 재구성 오차 (Reconstruction Error) → 클수록 이상
- **왜 필요한가**: 정상 거래 데이터만 학습하면, 이상 거래가 들어왔을 때
  재구성을 잘 못 해서 오차가 커집니다. 레이블(정답) 없이 이상치를 탐지할 수 있습니다.

### 모델 4 — K-Means (피어 그룹 군집화)

- **역할**: 비슷한 조건의 아파트끼리 묶어 또래 비교 기준 생성
- **입력**: 시군구, 전용면적 구간, 준공연도 구간, 세대수
- **출력**: 각 단지의 cluster_id
- **왜 필요한가**: 서울 강남 대형 아파트와 충남 소형 빌라를 같은 기준으로
  비교하면 의미가 없습니다. 같은 군집 내 거래 평균과 비교해야 합니다.
  Z-score = (실제가 - 군집평균) / 군집표준편차로 이탈 여부를 판정합니다.

### 모델 5 — KR-FinBERT (한국어 금융 감성 분석)

- **역할**: 뉴스 텍스트로 시장 분위기 점수화
- **모델**: `snunlp/KR-FinBert-SC` (Hugging Face, 이미 금융 텍스트로 학습 완료)
- **입력**: 해당 지역·단지 관련 네이버 뉴스 텍스트
- **출력**: 긍정 / 부정 / 중립 감성 점수
- **왜 필요한가**: 가격이 급등했을 때 "재건축 호재"같은 실제 이유가 뉴스에 없다면
  시세 조작 의심 근거가 됩니다. 별도 파인튜닝 없이 바로 사용 가능합니다.

### 모델 6 — Logistic Regression (로지스틱 회귀 분류)

- **역할**: S1~S4 네 점수를 받아 최종 위험도 분류
- **입력**: [S1(Autoencoder점수), S2(XGBoost 괴리율), S3(Z-score), S4(뉴스감성)]
- **출력**: 정상 / 주의 / 고위험
- **왜 필요한가**: 단순 규칙 `score > 0.7 이면 고위험` 보다 데이터로 학습한
  분류 경계가 더 신뢰성 있습니다. 인공 이상치 데이터로 학습합니다.

### 보조 — SHAP (설명 가능 AI)

- **역할**: "왜 이 거래가 이상인가?" 변수별 기여도 계산 및 시각화
- **연결**: XGBoost와 궁합이 좋아 빠르고 정확한 SHAP 값 계산 가능
- **출력**: 바 차트, 워터폴 차트로 변수 기여도 시각화

---

## 4. 기술 스택

| 분류 | 기술 | 사용 목적 |
|------|------|-----------|
| **Language** | Python 3.10+ | 전체 코드베이스 |
| **데이터 수집** | `requests`, `BeautifulSoup` | 국토부/K-apt API 호출 및 뉴스 크롤링 |
| **데이터 파이프라인** | `pandas`, n8n | CSV 전처리 병합, 당월 API 스케줄링 |
| **머신러닝** | `scikit-learn` | Linear Regression, Logistic Regression, K-Means, 피처 스케일링 |
| **그래디언트 부스팅** | `xgboost` | 적정 가격 회귀 예측 |
| **딥러닝** | `PyTorch` | Autoencoder 설계 및 학습 |
| **NLP** | `transformers` (KR-FinBERT) | 뉴스 감성 분석 (Hugging Face 사전학습 모델) |
| **XAI** | `shap` | 변수 기여도 시각화 |
| **백엔드** | `FastAPI`, `uvicorn` | 모델 추론 결과 REST API 제공 |
| **알림** | `python-telegram-bot` | 고위험 거래 텔레그램 실시간 알림 |

---

## 5. 디렉토리 구조

```
real-estate-ml/
│
├── data/                          # ⚠️ .gitignore 포함 — 절대 커밋 금지
│   ├── raw/                       # 공공데이터포털에서 다운받은 원본 CSV 파일
│   ├── processed/                 # 전처리 · 병합 완료된 학습용 데이터셋
│   └── external/                  # 금리, 통계청 등 보조 참고 데이터
│
├── docs/                          # 프로젝트 문서
│   └── DATA_STRATEGY.md           # 투트랙 데이터 수집 전략 상세 설명
│
├── src/                           # 핵심 소스코드 (여기서 작업합니다)
│   │
│   ├── data/                      # 데이터 수집 모듈
│   │   ├── molit_api.py           # 국토부 실거래가 당월 API 수집
│   │   ├── kapt_api.py            # K-apt 관리비 당월 API 수집
│   │   ├── news_crawler.py        # 네이버 뉴스 크롤링
│   │   └── batch_loader.py        # 과거 CSV 파일 일괄 로드 및 병합
│   │
│   ├── features/                  # 전처리 · 피처 엔지니어링
│   │   ├── preprocessor.py        # 결측치 처리, 이상값 제거, 타입 변환
│   │   └── integrator.py          # 단지 코드 기준 세 데이터 통합 테이블 생성
│   │
│   ├── models/                    # 모델 정의 · 학습 · 저장
│   │   ├── linear_regression.py   # [모델1] Linear Regression 학습 및 평가
│   │   ├── xgboost_regressor.py   # [모델2] XGBoost 가격 예측 및 괴리율 계산
│   │   ├── autoencoder.py         # [모델3] Autoencoder 설계 · 학습 · 오차 계산
│   │   ├── peer_group.py          # [모델4] K-Means 군집화 및 Z-score 계산
│   │   ├── finbert_sentiment.py   # [모델5] KR-FinBERT 뉴스 감성 분석
│   │   └── logistic_classifier.py # [모델6] Logistic Regression 최종 위험도 분류
│   │
│   ├── reasoning/                 # 점수 통합 · 판정 · 설명
│   │   ├── scorer.py              # S1~S4 점수 0~1 정규화
│   │   ├── triple_check.py        # 네 점수 → Logistic Regression 위험도 판정
│   │   └── shap_explainer.py      # SHAP 기여도 계산 및 차트 저장
│   │
│   ├── api/                       # 서비스 인터페이스
│   │   ├── main.py                # FastAPI 앱 진입점
│   │   ├── routes.py              # /analyze, /report 등 API 엔드포인트
│   │   └── notifier.py            # 텔레그램 봇 알림 발송 모듈
│   │
│   └── utils/                     # 공통 유틸리티
│       ├── config.py              # .env 환경 변수 로드
│       └── logger.py              # 로그 출력 설정
│
├── configs/                       # 하이퍼파라미터 설정 파일
│   ├── model_config.yaml          # 각 모델의 파라미터 (epochs, lr, k 등)
│   └── threshold_config.yaml      # 위험도 판정 임계값 설정
│
├── reports/                       # 모델 평가 결과 · 시각화 이미지 저장
│   ├── model_comparison.png       # Linear vs XGBoost 성능 비교 차트
│   └── shap_summary.png           # SHAP 변수 기여도 차트
│
├── notebooks/                     # 탐색적 데이터 분석(EDA) 주피터 노트북
│   └── eda.ipynb                  # 데이터 분포 시각화 · 상관관계 분석
│
├── scripts/                       # 한 번만 실행하는 오프라인 학습 스크립트
│   ├── train_all.py               # 모델 6개 순서대로 학습하고 저장
│   └── generate_synthetic.py      # 인공 이상치 데이터 생성 (Logistic 학습용)
│
├── GIT_GUIDE.md                   # 팀원용 Git 사용 설명서
├── .env.example                   # 환경 변수 템플릿 (이걸 복사해서 .env 만드세요)
├── .gitignore                     # Git 제외 목록
├── requirements.txt               # Python 패키지 목록
└── README.md                      # 본 문서
```

---

## 6. 역할 분담 및 작업 가이드

> 작업 전 반드시 `GIT_GUIDE.md` 를 먼저 읽고, 본인 브랜치를 만들어 작업하세요.

---

### 👑 문승준 (팀 리드) — ETL 파이프라인 + Autoencoder

#### 담당 파일
- `src/data/batch_loader.py`
- `src/features/preprocessor.py`
- `src/features/integrator.py`
- `src/models/autoencoder.py`
- `scripts/train_all.py`

#### 작업 순서

**① 과거 CSV 배치 로더 (`batch_loader.py`)**

공공데이터포털에서 다운받은 CSV 파일들을 읽어서 하나의 데이터프레임으로 합칩니다.

```python
# 예시 구조
import pandas as pd
import os

def load_molit_csvs(folder_path: str) -> pd.DataFrame:
    """data/raw/ 폴더 안의 국토부 CSV 파일을 전부 읽어 합칩니다."""
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dfs = [pd.read_csv(os.path.join(folder_path, f), encoding='cp949') for f in files]
    return pd.concat(dfs, ignore_index=True)
```

**② 전처리 (`preprocessor.py`)**

결측치, 이상값, 자료형 변환을 처리합니다.

```python
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['거래금액', '전용면적'])   # 필수 컬럼 결측 제거
    df['거래금액'] = df['거래금액'].str.replace(',', '').astype(float)
    df = df[df['거래금액'] > 0]                       # 0원 거래 제거
    return df
```

**③ 데이터 통합 (`integrator.py`)**

실거래가, 관리비, 뉴스 점수를 단지 코드 기준으로 LEFT JOIN 합니다.

```python
def integrate(molit_df, kapt_df, news_df) -> pd.DataFrame:
    df = molit_df.merge(kapt_df, on='단지코드', how='left')
    df = df.merge(news_df, on=['단지코드', '거래년월'], how='left')
    return df
```

**④ Autoencoder 모델 (`autoencoder.py`)**

PyTorch로 인코더-디코더 구조를 만들고, 정상 거래 데이터로만 학습합니다.

```python
import torch
import torch.nn as nn

class ApartmentAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 8)          # 압축
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)  # 복원
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def reconstruction_error(model, x_tensor):
    """재구성 오차 계산 — 클수록 이상 거래"""
    with torch.no_grad():
        pred = model(x_tensor)
        error = torch.mean((pred - x_tensor) ** 2, dim=1)
    return error.numpy()
```

**⑤ n8n 스케줄러 설정**

로컬에 n8n을 설치하고, 매일 새벽 2시에 당월 API 수집 스크립트를 자동 실행하도록
워크플로우를 구성합니다. 완료 시 텔레그램으로 "수집 완료" 알림을 보냅니다.

---

### 🌐 손종인 — API 수집 + Peer Group + 백엔드

#### 담당 파일
- `src/data/molit_api.py`
- `src/data/kapt_api.py`
- `src/data/news_crawler.py`
- `src/models/peer_group.py`
- `src/api/main.py`
- `src/api/routes.py`
- `src/api/notifier.py`

#### 작업 순서

**① 국토부 API 수집 (`molit_api.py`)**

`.env` 에 있는 키로 당월 실거래가를 가져옵니다.

```python
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def fetch_molit(lawd_cd: str, deal_ymd: str) -> list[dict]:
    """
    lawd_cd  : 지역코드 (예: 41135 = 성남시 분당구)
    deal_ymd : 거래년월 (예: 202504)
    """
    url = "http://openapi.molit.go.kr/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTrade"
    params = {
        'serviceKey': os.getenv('MOLIT_API_KEY'),
        'LAWD_CD': lawd_cd,
        'DEAL_YMD': deal_ymd,
        'numOfRows': 1000,
        'pageNo': 1,
    }
    response = requests.get(url, params=params)
    # XML 파싱 후 리스트로 반환
    ...
```

**② Peer Group 군집화 (`peer_group.py`)**

K-Means로 비슷한 아파트를 묶고, 같은 군집 내에서 Z-score를 계산합니다.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

def build_peer_groups(df, n_clusters=20):
    features = ['위도', '경도', '면적구간', '준공연도구간', '세대수']
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster_id'] = km.fit_predict(X_scaled)
    return df, km, scaler

def peer_zscore(df):
    """같은 클러스터 내 거래가 Z-score 계산"""
    df['peer_mean'] = df.groupby('cluster_id')['거래금액'].transform('mean')
    df['peer_std']  = df.groupby('cluster_id')['거래금액'].transform('std')
    df['peer_zscore'] = (df['거래금액'] - df['peer_mean']) / df['peer_std']
    return df
```

**③ FastAPI 백엔드 (`main.py`, `routes.py`)**

분석 결과를 외부에서 호출할 수 있도록 REST API로 노출합니다.

```python
# main.py
from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(title="부동산 이상 거래 탐지 API")
app.include_router(router)

# routes.py
from fastapi import APIRouter
router = APIRouter()

@router.get("/analyze/{단지코드}")
def analyze(단지코드: str):
    """단지 코드를 입력하면 위험도 분석 결과 반환"""
    ...
```

**④ 텔레그램 알림 (`notifier.py`)**

고위험 거래 탐지 시 팀 텔레그램 채널로 메시지를 발송합니다.

```python
import telegram
import os

async def send_alert(message: str):
    bot = telegram.Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
    await bot.send_message(chat_id='채널ID', text=message)
```

---

### 🧠 안우현 — KR-FinBERT + Logistic Regression + SHAP

#### 담당 파일
- `src/models/linear_regression.py`
- `src/models/xgboost_regressor.py`
- `src/models/logistic_classifier.py`
- `src/models/finbert_sentiment.py`
- `src/reasoning/scorer.py`
- `src/reasoning/triple_check.py`
- `src/reasoning/shap_explainer.py`
- `scripts/generate_synthetic.py`

#### 작업 순서

**① Linear Regression 베이스라인 (`linear_regression.py`)**

XGBoost와 비교용으로 먼저 학습합니다.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def train_linear(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"Linear R²: {r2_score(y_test, pred):.4f}")
    print(f"Linear RMSE: {np.sqrt(mean_squared_error(y_test, pred)):,.0f} 만원")
    return model
```

**② XGBoost 가격 예측 (`xgboost_regressor.py`)**

```python
from xgboost import XGBRegressor
import numpy as np

def train_xgboost(X_train, y_train, X_test, y_test):
    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
    return model

def compute_deviation_score(model, X, y_actual):
    """괴리율 계산: (실제가 - 예측가) / 예측가"""
    y_pred = model.predict(X)
    deviation = (y_actual - y_pred) / y_pred
    return deviation   # 음수: 저평가, 양수: 고평가
```

**③ KR-FinBERT 감성 분석 (`finbert_sentiment.py`)**

Hugging Face에서 모델을 다운받아 바로 사용합니다. (파인튜닝 불필요)

```python
from transformers import pipeline

def load_sentiment_model():
    return pipeline(
        "text-classification",
        model="snunlp/KR-FinBert-SC",   # 한국어 금융 감성 BERT
        device=-1                         # CPU 사용 (-1), GPU 있으면 0
    )

def score_news(texts: list[str], model) -> list[float]:
    """
    뉴스 텍스트 리스트 → 부정 감성 점수 리스트 (0~1, 클수록 부정)
    """
    results = model(texts, truncation=True, max_length=512)
    scores = []
    for r in results:
        if r['label'] == 'negative':
            scores.append(r['score'])
        elif r['label'] == 'positive':
            scores.append(1 - r['score'])
        else:
            scores.append(0.5)
    return scores
```

**④ 인공 이상치 생성 (`generate_synthetic.py`)**

Logistic Regression 학습을 위한 레이블 데이터를 만듭니다.

```python
import pandas as pd
import numpy as np

def generate_synthetic_anomalies(normal_df: pd.DataFrame, ratio=0.3):
    """
    정상 데이터를 복사해 가격을 ±40% 조작 → 이상치로 레이블링
    """
    n = int(len(normal_df) * ratio)
    anomaly = normal_df.sample(n).copy()
    anomaly['거래금액'] *= np.random.uniform(1.4, 2.0, size=n)  # 고가 이상치
    anomaly['label'] = 1   # 이상

    normal_df['label'] = 0  # 정상
    return pd.concat([normal_df, anomaly], ignore_index=True)
```

**⑤ Logistic Regression 분류기 (`logistic_classifier.py`)**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def train_logistic(X_train, y_train):
    """
    X_train: [[S1, S2, S3, S4], ...]  — 네 점수
    y_train: [0, 1, 2, ...]           — 0=정상, 1=주의, 2=고위험
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, multi_class='multinomial')
    model.fit(X_scaled, y_train)
    return model, scaler
```

**⑥ 점수 통합 및 판정 (`scorer.py`, `triple_check.py`)**

```python
# scorer.py — 각 점수를 0~1로 정규화
from sklearn.preprocessing import MinMaxScaler

def normalize_scores(scores_df):
    scaler = MinMaxScaler()
    return scaler.fit_transform(scores_df)

# triple_check.py — 최종 위험도 판정
def judge(s1, s2, s3, s4, logistic_model, scaler):
    X = scaler.transform([[s1, s2, s3, s4]])
    label = logistic_model.predict(X)[0]
    proba = logistic_model.predict_proba(X)[0]
    risk_map = {0: '🟢 정상', 1: '🟡 주의', 2: '🔴 고위험'}
    return risk_map[label], max(proba)
```

**⑦ SHAP 설명 (`shap_explainer.py`)**

```python
import shap
import matplotlib.pyplot as plt

def explain_xgboost(model, X, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.savefig('reports/shap_summary.png', bbox_inches='tight')
    print("SHAP 차트가 reports/shap_summary.png 에 저장되었습니다.")
```

---

## 7. 로컬 환경 설정

> 처음 세팅하는 팀원은 아래 순서대로 따라하세요.

### Step 1 — 저장소 클론

```bash
git clone https://github.com/msjoon0811/real-estateml.git
cd real-estateml
```

### Step 2 — Python 가상 환경 만들기

```bash
# 가상 환경 생성
python -m venv .venv

# 가상 환경 활성화
# Windows
.venv\Scripts\activate
# Mac / Linux
source .venv/bin/activate

# 활성화 확인 — 터미널 앞에 (.venv) 표시가 뜨면 성공
```

### Step 3 — 패키지 설치

```bash
pip install -r requirements.txt
```

> 설치가 오래 걸립니다. PyTorch, Transformers 등 큰 패키지가 포함되어 있습니다.
> 오류가 나면 팀장(문승준)에게 오류 메시지를 캡처해서 보내주세요.

### Step 4 — 환경 변수 설정

```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

`.env` 파일을 열고 본인이 발급받은 API 키를 입력합니다.

```
MOLIT_API_KEY=여기에_국토부_키_입력
KAPT_API_KEY=여기에_Kapt_키_입력
NAVER_CLIENT_ID=여기에_네이버_아이디_입력
NAVER_CLIENT_SECRET=여기에_네이버_시크릿_입력
TELEGRAM_BOT_TOKEN=여기에_텔레그램_토큰_입력
```

### Step 5 — 실행 확인

```bash
# 환경 변수 로드 테스트
python -c "from src.utils.config import load_config; print('설정 로드 성공')"

# FastAPI 서버 기동
uvicorn src.api.main:app --reload

# 브라우저에서 http://localhost:8000/docs 접속하면 API 문서 확인 가능
```

### Step 6 — 오프라인 학습 실행 (처음 한 번만)

```bash
# 1. 인공 이상치 데이터 생성
python scripts/generate_synthetic.py

# 2. 모델 전체 학습 및 저장
python scripts/train_all.py
```

---

## 8. 일정 (마일스톤)

| 기간 | 마일스톤 | 주요 산출물 |
|------|----------|------------|
| **5월 1주차** | 데이터 수집 모듈 완성, 통합 테이블 구축 | 통합 데이터셋 v1, EDA 리포트 (`notebooks/eda.ipynb`) |
| **5월 2주차** | Linear/XGBoost 가격 예측 + Autoencoder + Peer Group 베이스라인 | 모델 성능 비교표, 이상치 탐지 결과 v1 |
| **5월 3주차** | KR-FinBERT 감성 + Logistic 분류기 + SHAP 통합 | Triple Check 전체 파이프라인 동작 확인 |
| **5월 4주차** | FastAPI 완성, 텔레그램 알림, 최종 문서 정리 | 최종 시스템, 학술 발표 자료 |

> **전체 연구 기간**: 2026. 4. 30. ~ 2026. 5. 31.

---

## 9. 보안 주의사항

절대로 아래 항목을 Git 커밋에 포함하지 마세요.

- **`.env` 파일** — API 키가 들어있습니다. `.gitignore` 에 등록되어 있습니다.
- **`data/` 폴더 내 파일** — 용량이 크고 개인정보가 포함될 수 있습니다.
- **모델 가중치 파일 (`.pt`, `.pkl`)** — 용량이 크므로 직접 공유합니다.

> 실수로 API 키를 커밋했다면 즉시 키를 폐기(revoke)하고 팀장에게 알려주세요!

---

## 10. 팀 연락망

| 역할 | 이름 | 담당 | 연락처 |
|------|------|------|--------|
| **지도교수** | 이성철 교수님 | — | sungchul@sunmoon.ac.kr |
| **팀 리드** | 문승준 | ETL 파이프라인, Autoencoder | msjoon0811@naver.com |
| **팀원** | 손종인 | API 수집, Peer Group, 백엔드 | sonjong9720@gmail.com |
| **팀원** | 안우현 | 가격 예측 모델, KR-FinBERT, Logistic, SHAP | 0215woo@naver.com |
