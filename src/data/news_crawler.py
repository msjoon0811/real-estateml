"""
네이버 뉴스 검색 API 크롤러
============================
- 엔드포인트: https://openapi.naver.com/v1/search/news.json
- 인증: X-Naver-Client-Id, X-Naver-Client-Secret 헤더
- .env 파일에 NAVER_CLIENT_ID, NAVER_CLIENT_SECRET 필요

실행:
    python3 -m src.data.news_crawler
"""

import json
import os
import re
import sys

import requests
from dotenv import load_dotenv

# ── 설정 ──────────────────────────────────────────────
load_dotenv()

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
SEARCH_URL = "https://openapi.naver.com/v1/search/news.json"


def _strip_html(text: str) -> str:
    """HTML 태그 제거 및 특수문자 디코딩"""
    clean = re.sub(r"<[^>]+>", "", text)
    clean = clean.replace("&quot;", '"').replace("&amp;", "&")
    clean = clean.replace("&lt;", "<").replace("&gt;", ">")
    clean = clean.replace("&apos;", "'")
    return clean.strip()


def fetch_news(keyword: str, display: int = 10) -> list[str]:
    """
    네이버 뉴스 검색 API로 뉴스 제목 목록을 가져옵니다.

    Args:
        keyword: 검색 키워드 (예: "강남 아파트 매매")
        display: 가져올 뉴스 개수 (최대 100, 기본 10)

    Returns:
        뉴스 제목 문자열 리스트

    Raises:
        ValueError: API 키가 설정되지 않은 경우
        requests.RequestException: API 호출 실패 시
    """
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        raise ValueError(
            "네이버 API 키가 설정되지 않았습니다. "
            ".env 파일에 NAVER_CLIENT_ID와 NAVER_CLIENT_SECRET을 추가하세요."
        )

    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }

    params = {
        "query": keyword,
        "display": min(display, 100),
        "start": 1,
        "sort": "date",  # 최신순
    }

    print(f"[뉴스] 검색 중... 키워드: '{keyword}', 개수: {display}")

    resp = requests.get(SEARCH_URL, headers=headers, params=params, timeout=15)
    resp.raise_for_status()

    data = resp.json()
    items = data.get("items", [])

    titles = [_strip_html(item["title"]) for item in items]
    return titles


def fetch_news_detail(keyword: str, display: int = 10) -> list[dict]:
    """
    뉴스 제목뿐 아니라 상세 정보(제목, 링크, 요약, 날짜)를 반환합니다.

    Args:
        keyword: 검색 키워드
        display: 가져올 뉴스 개수

    Returns:
        [{"title": ..., "link": ..., "description": ..., "pubDate": ...}, ...]
    """
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        raise ValueError(
            "네이버 API 키가 설정되지 않았습니다. "
            ".env 파일에 NAVER_CLIENT_ID와 NAVER_CLIENT_SECRET을 추가하세요."
        )

    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }

    params = {
        "query": keyword,
        "display": min(display, 100),
        "start": 1,
        "sort": "date",
    }

    resp = requests.get(SEARCH_URL, headers=headers, params=params, timeout=15)
    resp.raise_for_status()

    data = resp.json()
    items = data.get("items", [])

    results = []
    for item in items:
        results.append({
            "title": _strip_html(item.get("title", "")),
            "link": item.get("originallink") or item.get("link", ""),
            "description": _strip_html(item.get("description", "")),
            "pubDate": item.get("pubDate", ""),
        })

    return results


def main():
    """부동산 관련 뉴스 검색 테스트"""
    keyword = "서울 아파트 매매"

    try:
        # 1) 기본 함수 — 제목만
        titles = fetch_news(keyword, display=5)
        print(f"\n📰 '{keyword}' 뉴스 제목 ({len(titles)}건):")
        print("-" * 50)
        for i, title in enumerate(titles, 1):
            print(f"  {i}. {title}")

        # 2) 상세 함수 — 전체 정보
        print(f"\n📰 '{keyword}' 뉴스 상세:")
        print("-" * 50)
        details = fetch_news_detail(keyword, display=5)
        print(json.dumps(details, ensure_ascii=False, indent=2))

    except ValueError as e:
        print(f"❌ 오류: {e}", file=sys.stderr)
        sys.exit(1)
    except requests.RequestException as e:
        print(f"❌ API 요청 실패: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n✅ 네이버 뉴스 API 호출 완료!")


if __name__ == "__main__":
    main()
