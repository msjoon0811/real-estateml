# XGBoost·Autoencoder 기반 부동산 저평가 매물 탐지 및 이상 거래 필터링 시스템

> SW중심대학 산학협력 프로젝트 (2026학년도 1학기 기업연계 프로젝트)
> 선문대학교 컴퓨터공학부 · 기계학습프로젝트

실거래가(수치)·관리비(물리)·뉴스(텍스트)를 결합해 부동산 시장의 자전거래·시세 조작·거버넌스 리스크를 탐지하고, 그 근거를 설명 가능한 형태로 제공하는 데이터 포렌식 시스템입니다.

---

## 프로젝트 개요

### 문제 정의
기존 부동산 플랫폼(네이버부동산·직방 등)은 신고된 실거래가를 수동적으로 나열할 뿐, 해당 거래가 실제로 일어난 거래인지(자전거래 여부), 단지 내부에 어떤 운영 리스크가 있는지를 검증할 데이터가 부족합니다. 그 결과 일반 투자자와 실거주자가 거래의 실체적 진위와 객관적 가치를 판단하기 어렵습니다.

### 해결 접근
본 시스템은 세 가지 이종 데이터를 교차 검증해 이상 거래를 판별합니다.

- **수치 이상치**: 실거래가 패턴에서 벗어나는 거래를 Autoencoder 재구성 오차로 탐지
- **물리적 실체 검증**: 관리비·에너지 사용량으로 "사람이 실제 살고 있는지" 확인
- **상황적 확증**: 뉴스 텍스트 마이닝으로 가격 변동을 정당화할 호재가 실제 존재하는지 검증

이 세 축을 결합한 **트리플 체크(Triple Check)** 구조가 본 프로젝트의 핵심 차별점입니다.

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Data Source (실시간 API 수집)                              │
│    국토부 실거래가 │ K-apt 관리비/에너지 │ 네이버 뉴스          │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. ETL Pipeline (n8n 기반 전처리·통합)                        │
│    단지 코드 기준 데이터프레임 통합                              │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Model Engine Layer                                       │
│    ┌──────────────────┐    ┌──────────────────┐             │
│    │  Autoencoder     │    │     KoBERT       │             │
│    │  → 수치 이상치    │    │   → 심리 가중치    │             │
│    └──────────────────┘    └──────────────────┘             │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Reasoning Layer (XAI)                                    │
│    트리플 체크: 수치 이상치 │ 심리 가중치 │ 실체 검증            │
│    SHAP 기반 변수 기여도 산출                                  │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Service Interface (FastAPI)                              │
│    웹 대시보드 (분석 리포트) │ 텔레그램 (실시간 알림)             │
└─────────────────────────────────────────────────────────────┘
```

### 모델 구성 (Triple Model Engine)

| 모델 | 알고리즘 | 역할 |
|------|----------|------|
| Peer Group Clusterer | K-Means / KNN | 지리·세대수·준공연도 기반 또래 단지 군집화 |
| Anomaly Detection Engine | Autoencoder | 정상 거래 패턴 학습 및 재구성 오차 기반 이상치 판별 |
| Contextual Weighting Engine | KoBERT | 뉴스 감성 분석으로 시장 상황 가중치 산출 |
| XAI & Reasoning Engine | SHAP | 변수별 기여도 정량화 및 판정 근거 생성 |

---

## 기술 스택

- **Language**: Python 3.10+
- **ML/DL**: PyTorch, scikit-learn, XGBoost, Transformers (KoBERT), SHAP
- **Data Pipeline**: n8n, Pandas
- **Backend**: FastAPI
- **Notification**: Telegram Bot API
- **Data Source API**: 국토부 공공데이터포털, K-apt, 네이버 뉴스 API

---

## 디렉토리 구조 (제안)

```
real-estate-ml/
├── data/                      # 원본 및 가공 데이터 (gitignore)
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/                 # 실험·EDA용 Jupyter 노트북
├── src/
│   ├── data/                  # 데이터 수집·전처리 모듈
│   │   ├── molit_api.py       # 국토부 실거래가
│   │   ├── kapt_api.py        # K-apt 관리비
│   │   └── news_crawler.py    # 뉴스 수집
│   ├── features/              # 피처 엔지니어링
│   ├── models/
│   │   ├── peer_group.py      # K-Means / KNN 군집화
│   │   ├── autoencoder.py     # 이상치 탐지
│   │   ├── kobert_sentiment.py # 뉴스 감성 분석
│   │   └── shap_explainer.py  # XAI
│   ├── reasoning/             # 트리플 체크 로직
│   ├── api/                   # FastAPI 엔드포인트
│   └── utils/
├── tests/
├── configs/                   # 모델·파이프라인 설정
├── reports/                   # 분석 리포트·시각화 결과물
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 시작하기

### 1. 저장소 클론
```bash
git clone https://github.com/msjoon0811/real-estate-ml.git
cd real-estate-ml
```

### 2. 가상 환경 설정
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 환경 변수 설정
`.env.example` 파일을 복사해 `.env`로 만들고 본인의 API 키를 채워넣습니다.

```bash
cp .env.example .env
```

```env
# .env (절대 커밋 금지)
MOLIT_API_KEY=your_key_here
KAPT_API_KEY=your_key_here
NAVER_CLIENT_ID=your_id_here
NAVER_CLIENT_SECRET=your_secret_here
TELEGRAM_BOT_TOKEN=your_token_here
ANTHROPIC_API_KEY=your_key_here
```

### 4. 실행
```bash
# 데이터 수집
python -m src.data.molit_api --region 서울 --period 202601

# 모델 학습
python -m src.models.autoencoder --config configs/autoencoder.yaml

# API 서버 기동
uvicorn src.api.main:app --reload
```

---

## 협업 규칙

### 브랜치 전략 (Git Flow 간소화 버전)
- `main`: 배포 가능한 안정 버전. 직접 푸시 금지, PR을 통해서만 머지
- `develop`: 개발 통합 브랜치. 기능이 완성되면 여기로 머지
- `feature/{이슈번호}-{간단한-설명}`: 개별 기능 개발 (예: `feature/12-autoencoder-training`)
- `fix/{이슈번호}-{간단한-설명}`: 버그 수정
- `docs/{설명}`: 문서 작업

### 커밋 메시지 컨벤션
[Conventional Commits](https://www.conventionalcommits.org/) 형식을 따릅니다.

```
<type>: <subject>

<body (선택)>
```

| Type | 용도 |
|------|------|
| `feat` | 새로운 기능 추가 |
| `fix` | 버그 수정 |
| `docs` | 문서 수정 |
| `style` | 코드 포맷팅, 세미콜론 등 (로직 변경 X) |
| `refactor` | 코드 리팩토링 |
| `test` | 테스트 코드 추가·수정 |
| `chore` | 빌드·패키지 설정 등 기타 |
| `data` | 데이터 파이프라인·전처리 관련 |
| `model` | 모델 학습·튜닝 관련 |

예시:
```
feat: Autoencoder 재구성 오차 임계값 동적 산출 로직 추가
fix: K-apt API 응답에서 단지코드 누락 시 KeyError 처리
model: KoBERT 파인튜닝 epoch 5→10 조정
```

### Pull Request 규칙
- PR 제목은 커밋 컨벤션과 동일하게 작성
- 본문에 **무엇을·왜·어떻게** 변경했는지 기재
- 최소 1명 이상의 리뷰 승인 후 머지
- 머지는 가급적 **Squash and Merge** 사용

### 코드 스타일
- Python: [PEP 8](https://peps.python.org/pep-0008/) 준수
- 포매터: `black`, 린터: `ruff`
- 함수·클래스에는 docstring 작성 (Google 스타일)
- 모든 머신러닝 실험은 노트북이 아닌 **재현 가능한 스크립트**로 작성

### 이슈 관리
- 작업 시작 전 GitHub Issues에 등록
- 라벨: `data`, `model`, `api`, `bug`, `docs`, `discussion`
- 마일스톤 단위로 진행 상황 추적

---

## 역할 분담 (초안)

| 이름 | 학번 | 주요 담당 |
|------|------|-----------|
| 문승준 | 2022243031 | 팀 리드, Autoencoder·이상치 탐지, ETL 파이프라인 |
| 손종인 | 2022380043 | KoBERT 뉴스 감성 분석, XAI(SHAP) |
| 안우현 | 2022243007 | 데이터 수집(API), Peer Group 군집화, 서비스 인터페이스 |

> 역할은 진행 상황에 따라 유연하게 조정합니다. 매주 정기 미팅에서 재배분 여부를 점검합니다.

---

## 일정 (마일스톤)

| 기간 | 마일스톤 | 산출물 |
|------|----------|--------|
| 3월 | 데이터 수집·EDA, 단지 코드 기준 통합 테이블 구축 | 통합 데이터셋 v1, EDA 리포트 |
| 4월 | Peer Group 군집화 + Autoencoder 베이스라인 | 이상치 탐지 모델 v1 |
| 5월 | KoBERT 감성 분석 + SHAP 통합, FastAPI 인터페이스 | 트리플 체크 통합 모델 |
| 6월 | 대시보드·텔레그램 알림 완성, 발표 자료 정리 | 최종 시스템, 학술발표 자료 |

연구 기간: **2026. 3. 3. ~ 2026. 6. 19.**

---

## 보안 주의사항

다음 항목은 **절대로** 코드·문서·이슈·커밋 메시지에 포함하지 마세요.

- API 키 (국토부, K-apt, 네이버, Anthropic 등)
- GitHub Personal Access Token
- 텔레그램 봇 토큰
- 개인 연락처·주민번호 등 식별 정보
- `.env` 파일 자체

`.gitignore`에 다음 항목이 포함되어 있는지 확인하세요.

```gitignore
.env
.env.*
!.env.example
*.key
*.pem
data/raw/
data/processed/
__pycache__/
.venv/
.ipynb_checkpoints/
```

만약 실수로 시크릿을 커밋했다면, 단순 삭제로는 부족합니다. 즉시 해당 토큰·키를 **폐기(revoke)** 하고 새로 발급받아야 합니다. 필요시 `git filter-repo`로 히스토리에서 완전히 제거합니다.

---

## 참고문헌

- 국토교통부, 아파트매매 실거래 상세 자료, 공공데이터포털, https://www.data.go.kr
- 통계청, 장래인구추계, https://kosis.kr
- 한국은행, 경제통계시스템(ECOS) 기준금리, https://ecos.bok.or.kr
- K-apt 공동주택관리정보시스템, https://www.k-apt.go.kr

---

## 팀

| 역할 | 이름 | 연락 |
|------|------|------|
| 지도교수 | 이성철 교수님 | sungchul@sunmoon.ac.kr |
| 팀 리드 | 문승준 | msjoon0811@naver.com |
| 팀원 | 손종인 | sonjong9720@gmail.com |
| 팀원 | 안우현 | 0215woo@naver.com |

선문대학교 컴퓨터공학부 · SW중심대학사업단
