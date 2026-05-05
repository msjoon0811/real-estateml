"""
단지명 목록으로 네이버 뉴스 검색 → data/raw/news_data.csv 저장
"""

import os
import re
import time
import pandas as pd
import requests
from dotenv import load_dotenv

from src.models.finbert_sentiment import load_sentiment_model, score_news

load_dotenv()

NAVER_CLIENT_ID     = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
SEARCH_URL          = "https://openapi.naver.com/v1/search/news.json"
DATA_PATH           = "data/processed/integrated_dataset_v1_with_s1.csv"


def strip_html(text: str) -> str:
    clean = re.sub(r"<[^>]+>", "", text)
    for src, dst in [("&quot;", '"'), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&apos;", "'")]:
        clean = clean.replace(src, dst)
    return clean.strip()


def fetch_news(keyword: str, display: int = 5) -> list[dict]:
    headers = {
        "X-Naver-Client-Id":     NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {"query": keyword + " 아파트", "display": display, "start": 1, "sort": "date"}
    try:
        resp = requests.get(SEARCH_URL, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [{
            "단지명":      keyword,
            "title":       strip_html(item.get("title", "")),
            "description": strip_html(item.get("description", "")),
            "pubDate":     item.get("pubDate", ""),
            "link":        item.get("originallink") or item.get("link", ""),
        } for item in items]
    except Exception as e:
        print(f"  '{keyword}' 검색 실패: {e}")
        return []


def main():
    df        = pd.read_csv(DATA_PATH, encoding="utf-8-sig", low_memory=False)
    df_notna  = df[df["단지코드"].notna()]
    apt_names = df_notna["단지명"].dropna().unique().tolist()
    print(f"총 단지 수: {len(apt_names)}개")

    print("\n감성 분석 모델 로딩 중...")
    model = load_sentiment_model(verbose=False)
    print("모델 로딩 완료!\n")

    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        print(".env에 NAVER_CLIENT_ID, NAVER_CLIENT_SECRET 없음 — 샘플 데이터로 대체")
        sample_rows = [{
            "단지명":          name,
            "title":           f"[샘플] {name} 관련 뉴스",
            "description":     "API 키 설정 후 실제 데이터로 교체 필요",
            "pubDate":         "",
            "link":            "",
            "sentiment_score": 0.5,
        } for name in apt_names[:10]]
        result_df = pd.DataFrame(sample_rows)
        os.makedirs("data/raw", exist_ok=True)
        result_df.to_csv("data/raw/news_data.csv", index=False, encoding="utf-8-sig")
        print(f"샘플 CSV 저장: data/raw/news_data.csv ({len(result_df)}행)")
        return

    all_results = []
    for i, name in enumerate(apt_names):
        print(f"[{i+1}/{len(apt_names)}] '{name}' 검색 중...")
        results = fetch_news(name, display=5)

        if results:
            news_texts = [item["title"] + " " + item["description"] for item in results]
            scores     = score_news(news_texts, model)
            for item, score in zip(results, scores):
                item["sentiment_score"] = round(score, 4)

        all_results.extend(results)
        time.sleep(0.12)

        if (i + 1) % 100 == 0:
            os.makedirs("data/raw", exist_ok=True)
            pd.DataFrame(all_results).to_csv("data/raw/news_data.csv", index=False, encoding="utf-8-sig")
            print(f"  중간 저장 완료 ({len(all_results)}건)")

    os.makedirs("data/raw", exist_ok=True)
    pd.DataFrame(all_results).to_csv("data/raw/news_data.csv", index=False, encoding="utf-8-sig")
    print(f"\n완료! 총 {len(all_results)}건 → data/raw/news_data.csv 저장")


if __name__ == "__main__":
    main()
