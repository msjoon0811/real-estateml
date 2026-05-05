"""
finbert_sentiment.py
--------------------
KR-FinBERT (snunlp/KR-FinBert-SC) 기반 한국어 금융 뉴스 감성 분석

담당: 안우현

모델 출처: https://huggingface.co/snunlp/KR-FinBert-SC
  - 긍정(positive) / 부정(negative) / 중립(neutral) 3-class 분류
  - 파인튜닝 불필요, Hugging Face에서 직접 로드
  - 최초 실행 시 모델 자동 다운로드 (~400 MB)

사용 방법
─────────
from src.models.finbert_sentiment import load_sentiment_model, score_news

model  = load_sentiment_model()                  # 최초 한 번만 로드
scores = score_news(["뉴스 헤드라인..."], model)  # 0~1 부정 점수
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
import numpy as np

# transformers 일부 경고 억제
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ──────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────

MODEL_NAME  = "snunlp/KR-FinBert-SC"
REPORT_DIR  = "reports"

# 라벨 정규화 테이블 (모델 출력 라벨 → 통일 문자열)
_LABEL_MAP: Dict[str, str] = {
    "negative": "negative",
    "neg":      "negative",
    "NEGATIVE": "negative",
    "positive": "positive",
    "pos":      "positive",
    "POSITIVE": "positive",
    "neutral":  "neutral",
    "neu":      "neutral",
    "NEUTRAL":  "neutral",
}

# ── 키워드 보정 사전 (모델 예측 보완) ─────────────────────────
KEYWORD_BOOST = {
    "호재": [
        "규제 완화", "금리 인하", "LTV 완화", "취득세 인하", "취득세 감면",
        "재건축 허용", "공급 확대", "개발 호재", "GTX", "역세권",
        "상승", "급등", "회복", "활성화", "기대감", "완화 패키지",
    ],
    "악재": [
        "공급 부족", "전세난", "부도", "PF 부실", "미분양", "역전세",
        "깡통전세", "경매", "침체", "하락", "위기", "우려", "급감",
        "금리 인상", "규제 강화", "자금난", "부실",
    ],
}

def _apply_keyword_boost(text: str, base_negativity: float) -> float:
    """텍스트 내 호재/악재 키워드를 기반으로 부정 점수(0~1)를 보정."""
    boost = 0.0
    for kw in KEYWORD_BOOST["악재"]:
        if kw in text:
            boost += 0.15
    for kw in KEYWORD_BOOST["호재"]:
        if kw in text:
            boost -= 0.15
    return float(np.clip(base_negativity + boost, 0.0, 1.0))


# ──────────────────────────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────────────────────────

def load_sentiment_model(device: int = -1, verbose: bool = True):
    """
    KR-FinBERT 감성 분류 파이프라인 로드.

    Parameters
    ----------
    device  : -1=CPU (기본), 0 이상=GPU 번호
    verbose : 진행 메시지 출력 여부

    Returns
    -------
    pipeline — Hugging Face text-classification 파이프라인
    """
    from transformers import pipeline

    if verbose:
        print(f"  KR-FinBERT 로드 중... (model: {MODEL_NAME})")
        print("  ※ 첫 실행 시 Hugging Face에서 ~400MB 다운로드됩니다.")

    t0    = time.time()
    model = pipeline(
        "text-classification",
        model=MODEL_NAME,
        device=device,
        truncation=True,
        max_length=512,
        return_all_scores=False,   # 최고 확률 1개만 반환
    )
    elapsed = time.time() - t0

    if verbose:
        dev_str = f"GPU:{device}" if device >= 0 else "CPU"
        print(f"  KR-FinBERT 로드 완료 !  ({dev_str}, {elapsed:.1f}초)")

    return model


# ──────────────────────────────────────────────────────────────
# 감성 점수 변환
# ──────────────────────────────────────────────────────────────

def _label_to_negativity(label: str, score: float) -> float:
    """
    단일 분류 결과 → 부정 확률 (0~1).

    negative : score 그대로   (부정 확률)
    positive : 1 - score      (부정 방향으로 반전)
    neutral  : 0.5            (중립)
    """
    normalized = _LABEL_MAP.get(label, "neutral")
    if normalized == "negative":
        return float(score)
    elif normalized == "positive":
        return float(1.0 - score)
    else:
        return 0.5


# ──────────────────────────────────────────────────────────────
# 단일 / 리스트 점수화
# ──────────────────────────────────────────────────────────────

def score_news(texts: List[str], model) -> List[float]:
    """
    뉴스 텍스트 리스트 → 부정 감성 점수 리스트 (0~1, 클수록 부정).

    Parameters
    ----------
    texts : 분석할 뉴스 헤드라인 또는 본문 리스트
    model : load_sentiment_model() 로 반환된 파이프라인

    Returns
    -------
    List[float] — 각 텍스트의 부정 감성 점수 (0~1)

    Notes
    -----
    - 빈 리스트 입력 시 빈 리스트 반환
    - 빈 문자열은 중립(0.5)으로 처리
    """
    if not texts:
        return []

    # 빈 문자열 인덱스 따로 처리
    valid_idx   = [i for i, t in enumerate(texts) if t and t.strip()]
    neutral_idx = [i for i, t in enumerate(texts) if not (t and t.strip())]

    scores: List[float] = [0.5] * len(texts)

    if valid_idx:
        valid_texts = [texts[i] for i in valid_idx]
        results     = model(valid_texts, truncation=True, max_length=512)
        for i, r, text in zip(valid_idx, results, valid_texts):
            base_score = _label_to_negativity(r["label"], r["score"])
            scores[i] = _apply_keyword_boost(text, base_score)

    return scores


def score_single(text: str, model) -> Tuple[float, str, float]:
    """
    단일 텍스트 분석.

    Returns
    -------
    (negativity_score, raw_label, raw_score)
        negativity_score : 부정 확률 (0~1)
        raw_label        : 모델 원본 라벨
        raw_score        : 모델 원본 확률
    """
    result     = model([text], truncation=True, max_length=512)[0]
    base_score = _label_to_negativity(result["label"], result["score"])
    neg_score  = _apply_keyword_boost(text, base_score)
    return neg_score, result["label"], result["score"]


# ──────────────────────────────────────────────────────────────
# 배치 처리 (대량 뉴스 처리용)
# ──────────────────────────────────────────────────────────────

def score_news_batch(
    texts:      List[str],
    model,
    batch_size: int  = 32,
    verbose:    bool = True,
) -> List[float]:
    """
    메모리 효율적인 배치 단위 감성 점수 계산.

    Parameters
    ----------
    texts      : 뉴스 텍스트 리스트
    model      : KR-FinBERT 파이프라인
    batch_size : 배치 크기 (기본 32, GPU 사용 시 64~128 권장)
    verbose    : 진행 상황 출력 여부

    Returns
    -------
    List[float] — 부정 감성 점수 리스트
    """
    if not texts:
        return []

    all_scores: List[float] = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch  = texts[i : i + batch_size]
        scores = score_news(batch, model)
        all_scores.extend(scores)
        if verbose:
            done = min(i + batch_size, total)
            pct  = done / total * 100
            bar  = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"\r  감성 분석 [{bar}] {done}/{total} ({pct:.0f}%)", end="", flush=True)

    if verbose:
        print()  # 줄 바꿈

    return all_scores


# ──────────────────────────────────────────────────────────────
# 통계 요약
# ──────────────────────────────────────────────────────────────

def summarize_sentiment(scores: List[float]) -> Dict[str, float]:
    """
    감성 점수 리스트의 통계 요약.

    Returns
    -------
    dict
        {'mean', 'std', 'min', 'max', 'pct_negative', 'pct_positive', 'pct_neutral'}
    """
    arr = np.array(scores)
    return {
        "mean":         float(arr.mean()),
        "std":          float(arr.std()),
        "min":          float(arr.min()),
        "max":          float(arr.max()),
        "pct_negative": float((arr > 0.6).mean() * 100),   # 부정 비율 (%)
        "pct_positive": float((arr < 0.4).mean() * 100),   # 긍정 비율 (%)
        "pct_neutral":  float(((arr >= 0.4) & (arr <= 0.6)).mean() * 100),
    }


# ──────────────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────────────

def plot_sentiment_distribution(
    scores:  List[float],
    title:   str = "뉴스 감성 점수 분포",
    fname:   str = "sentiment_distribution.png",
) -> None:
    """
    감성 점수 히스토그램 + KDE 저장 (reports/<fname>).

    Parameters
    ----------
    scores : score_news() 또는 score_news_batch() 결과
    title  : 차트 제목
    fname  : 저장 파일명
    """
    os.makedirs(REPORT_DIR, exist_ok=True)

    arr = np.array(scores)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 히스토그램
    ax = axes[0]
    ax.hist(arr, bins=30, color="#C44E52", edgecolor="white", alpha=0.8)
    ax.axvline(0.4, color="#55A868", linestyle="--", linewidth=1.5, label="긍정 경계 (0.4)")
    ax.axvline(0.6, color="#4C72B0", linestyle="--", linewidth=1.5, label="부정 경계 (0.6)")
    ax.set_xlabel("부정 감성 점수 (0~1)")
    ax.set_ylabel("빈도")
    ax.set_title(title)
    ax.legend(fontsize=8)

    # 파이 차트
    summary = summarize_sentiment(scores)
    ax2 = axes[1]
    labels_ = ["부정 (>0.6)", "긍정 (<0.4)", "중립"]
    sizes_  = [summary["pct_negative"], summary["pct_positive"], summary["pct_neutral"]]
    colors_ = ["#C44E52", "#55A868", "#8172B2"]
    ax2.pie(sizes_, labels=labels_, colors=colors_, autopct="%1.1f%%", startangle=90)
    ax2.set_title("감성 비율")

    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, fname)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  감성 분포 차트 저장 → {save_path}")


# ──────────────────────────────────────────────────────────────
# 단독 실행 테스트
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    실행:
        python -m src.models.finbert_sentiment
    또는:
        python src/models/finbert_sentiment.py
    """
    print("\n" + "=" * 52)
    print("  KR-FinBERT 감성 분석 단독 테스트")
    print("=" * 52)

    sample_texts = [
        "분당 재건축 호재 발표로 집값 급등 예상",
        "A단지 주민 소송으로 관리비 분쟁 심화",
        "금리 인상으로 부동산 거래 위축"
    ]

    # 모델 로드
    mdl = load_sentiment_model(device=-1, verbose=True)

    # 배치 점수화
    print("\n  뉴스 감성 분석 중...")
    scores = score_news_batch(sample_texts, mdl, batch_size=4, verbose=True)

    # 결과 출력
    print("\n  [감성 분석 결과]")
    for text, score in zip(sample_texts, scores):
        bar   = "█" * int(score * 20)
        label = "🔴부정" if score > 0.6 else ("🟢긍정" if score < 0.4 else "⚪중립")
        print(f"  {score:.3f} |{bar:<20}| {label}  {text[:35]}...")

    # 통계 요약
    summary = summarize_sentiment(scores)
    print("\n  [통계 요약]")
    for k, v in summary.items():
        print(f"    {k:<16}: {v:.3f}")

    # 시각화
    plot_sentiment_distribution(scores, title="샘플 뉴스 감성 점수 분포")

    # 단일 텍스트 상세 분석
    print("\n  [단일 텍스트 상세]")
    neg_score, raw_label, raw_score = score_single(sample_texts[0], mdl)
    print(f"    텍스트  : {sample_texts[0][:45]}...")
    print(f"    원본 라벨: {raw_label}  원본 확률: {raw_score:.4f}")
    print(f"    부정 점수: {neg_score:.4f}")

    print("\n  테스트 완료 !  →  reports/ 폴더를 확인하세요.")
