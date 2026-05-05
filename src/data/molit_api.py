"""
국토교통부 아파트 매매 실거래가 API 호출 모듈
==============================================
- 엔드포인트: http://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev
- 필수 파라미터: serviceKey, LAWD_CD (법정동코드 5자리), DEAL_YMD (계약년월 YYYYMM)
- .env 파일에 MOLIT_API_KEY 필요

실행:
    python3 -m src.data.molit_api
"""

import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

import requests
from dotenv import load_dotenv

# ── 설정 ──────────────────────────────────────────────
load_dotenv()

API_KEY = os.getenv("MOLIT_API_KEY")
BASE_URL = (
    "http://apis.data.go.kr/1613000/"
    "RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev"
)

# 서울 주요 구 법정동코드 (앞 5자리)
DEFAULT_LAWD_CODES = {
    "11110": "종로구",
    "11140": "중구",
    "11170": "용산구",
    "11200": "성동구",
    "11215": "광진구",
    "11230": "동대문구",
    "11260": "중랑구",
    "11290": "성북구",
    "11305": "강북구",
    "11320": "도봉구",
    "11350": "노원구",
    "11380": "은평구",
    "11410": "서대문구",
    "11440": "마포구",
    "11470": "양천구",
    "11500": "강서구",
    "11530": "구로구",
    "11545": "금천구",
    "11560": "영등포구",
    "11590": "동작구",
    "11620": "관악구",
    "11650": "서초구",
    "11680": "강남구",
    "11710": "송파구",
    "11740": "강동구",
}


def fetch_apt_trade(
    lawd_cd: str = "11680",
    deal_ymd: str | None = None,
    num_of_rows: int = 100,
    page_no: int = 1,
) -> dict:
    """
    국토교통부 아파트 매매 실거래가 데이터를 조회합니다.

    Args:
        lawd_cd: 법정동코드 앞 5자리 (기본값: 강남구 11680)
        deal_ymd: 계약년월 YYYYMM (기본값: 당월)
        num_of_rows: 한 페이지 결과 수
        page_no: 페이지 번호

    Returns:
        API 응답 JSON dict
    """
    if not API_KEY:
        raise ValueError(
            "MOLIT_API_KEY가 설정되지 않았습니다. "
            ".env 파일에 MOLIT_API_KEY=... 를 추가하세요."
        )

    if deal_ymd is None:
        deal_ymd = datetime.now().strftime("%Y%m")

    params = {
        "serviceKey": API_KEY,
        "LAWD_CD": lawd_cd,
        "DEAL_YMD": deal_ymd,
        "numOfRows": num_of_rows,
        "pageNo": page_no,
        "type": "json",
    }

    print(f"[MOLIT] 요청 중... 지역: {lawd_cd}, 계약년월: {deal_ymd}")

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()

    # JSON 응답 시도
    try:
        data = resp.json()
        return data
    except (requests.exceptions.JSONDecodeError, ValueError):
        pass

    # XML 응답 파싱
    try:
        root = ET.fromstring(resp.text)
        # 결과코드 확인
        result_code = root.findtext(".//resultCode", "")
        result_msg = root.findtext(".//resultMsg", "")
        total_count = root.findtext(".//totalCount", "0")

        items = []
        for item in root.iter("item"):
            row = {}
            for child in item:
                row[child.tag] = (child.text or "").strip()
            items.append(row)

        return {
            "resultCode": result_code,
            "resultMsg": result_msg,
            "totalCount": int(total_count),
            "items": items,
        }
    except ET.ParseError:
        print("[MOLIT] XML 파싱도 실패. 원본 응답:")
        print(resp.text[:2000])
        return {"error": "파싱 실패", "raw": resp.text[:2000]}


def main():
    """강남구 아파트 실거래가 데이터 조회 (테스트)"""
    try:
        # 당월 시도 → 데이터 없으면 직전월
        data = fetch_apt_trade(deal_ymd=datetime.now().strftime("%Y%m"))
        if data.get("totalCount", 0) == 0:
            from datetime import timedelta
            prev = (datetime.now().replace(day=1) - timedelta(days=1))
            prev_ymd = prev.strftime("%Y%m")
            print(f"\n[MOLIT] 당월 데이터 없음. 직전월({prev_ymd}) 재시도...")
            data = fetch_apt_trade(deal_ymd=prev_ymd)
    except ValueError as e:
        print(f"❌ 오류: {e}", file=sys.stderr)
        sys.exit(1)
    except requests.RequestException as e:
        print(f"❌ API 요청 실패: {e}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(data, ensure_ascii=False, indent=2))
    count = data.get("totalCount", len(data.get("items", [])))
    print(f"\n✅ 국토교통부 실거래가 API 호출 완료! (총 {count}건)")


if __name__ == "__main__":
    main()
