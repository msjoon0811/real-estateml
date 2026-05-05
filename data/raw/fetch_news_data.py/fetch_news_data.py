"""
단지명 목록으로 네이버 뉴스 검색 → data/raw/news_data.csv 저장
"""

import os
import re
import time
import pandas as pd
import requests
from dotenv import load_dotenv

import sys
# 프로젝트 루트 경로(real-estateml)를 파이썬 모듈 검색 경로에 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PROJECT_ROOT)

from src.models.finbert_sentiment import load_sentiment_model, score_news

load_dotenv()

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
SEARCH_URL = "https://openapi.naver.com/v1/search/news.json"


def strip_html(text: str) -> str:
    clean = re.sub(r"<[^>]+>", "", text)
    for src, dst in [("&quot;", '"'), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&apos;", "'")]:
        clean = clean.replace(src, dst)
    return clean.strip()


def fetch_news(keyword: str, display: int = 5) -> list[dict]:
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {"query": keyword + " 아파트", "display": display, "start": 1, "sort": "date"}
    try:
        resp = requests.get(SEARCH_URL, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [{
            "단지명": keyword,
            "title": strip_html(item.get("title", "")),
            "description": strip_html(item.get("description", "")),
            "pubDate": item.get("pubDate", ""),
            "link": item.get("originallink") or item.get("link", ""),
        } for item in items]
    except Exception as e:
        print(f"  ⚠️  '{keyword}' 검색 실패: {e}")
        return []


def main():
    # 데이터 로드
    df = pd.read_csv("data/raw/integrated_dataset_v1.csv", encoding="utf-8-sig")
    df_notna = df[df["단지코드"].notna()]
    apt_names = df_notna["단지명"].dropna().unique().tolist()
    print(f"총 단지 수: {len(apt_names)}개")

    # [연동] 사용자님의 감성 분석 AI 모델 미리 준비
    print("\n[AI] 감성 분석 모델을 불러오는 중입니다 (최초 1회)...")
    model = load_sentiment_model(verbose=False)
    print("[AI] 모델 로딩 완료!\n")

    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        print("❌ .env에 NAVER_CLIENT_ID, NAVER_CLIENT_SECRET 없음 — 샘플 데이터로 대체")
        # API 키 없을 때 샘플 CSV 생성 (구조 확인용)
        sample_rows = []
        for name in apt_names[:10]:
            sample_rows.append({
                "단지명": name,
                "title": f"[샘플] {name} 관련 뉴스",
                "description": "API 키 설정 후 실제 데이터로 교체 필요",
                "pubDate": "",
                "link": "",
                "sentiment_score": 0.5,  # 샘플 점수 추가
            })
        result_df = pd.DataFrame(sample_rows)
        os.makedirs("data/raw", exist_ok=True)
        result_df.to_csv("data/raw/news_data.csv", index=False, encoding="utf-8-sig")
        print(f"샘플 CSV 저장 완료: data/raw/news_data.csv ({len(result_df)}행)")
        return

    # 실제 API 호출
    all_results = []
    for i, name in enumerate(apt_names):
        print(f"[{i+1}/{len(apt_names)}] '{name}' 검색 중...")
        results = fetch_news(name, display=5)
        
        # [연동] 가져온 뉴스가 있다면, 즉시 감성 점수를 계산해서 딕셔너리에 추가
        if results:
            # 1. 텍스트 추출 (제목 + 본문)
            news_texts = [item["title"] + " " + item["description"] for item in results]
            
            # 2. 감성 점수 획득 (사용자님의 함수 사용)
            scores = score_news(news_texts, model)
            
            # 3. 딕셔너리에 'sentiment_score' 항목 추가
            for item, score in zip(results, scores):
                item["sentiment_score"] = round(score, 4)

        all_results.extend(results)
        # API 과부하 방지 (초당 10건 제한)
        time.sleep(0.12)

        # 중간 저장 (100개마다)
        if (i + 1) % 100 == 0:
            temp_df = pd.DataFrame(all_results)
            os.makedirs("data/raw", exist_ok=True)
            temp_df.to_csv("data/raw/news_data.csv", index=False, encoding="utf-8-sig")
            print(f"  → 중간 저장 완료 ({len(all_results)}건)")

    # 최종 저장
    result_df = pd.DataFrame(all_results)
    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/news_data.csv"
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 완료! 총 {len(result_df)}건 → {output_path} 저장")


if __name__ == "__main__":
    main()
