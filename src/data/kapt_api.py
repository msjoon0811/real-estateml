"""
K-apt 공동주택 정보 API 호출 모듈
==================================
공공데이터포털 API를 사용합니다:
  ① 공동주택 단지 목록제공 서비스  → 시도코드로 단지 목록(kaptCode) 조회
  ② 공동주택 기본 정보제공 서비스  → kaptCode로 단지 기본정보 조회

- .env 파일에 KAPT_API_KEY 필요 (Decoding 키 사용)

실행:
    python3 -m src.data.kapt_api
"""

import json
import os
import sys

import requests
from dotenv import load_dotenv

# ── 설정 ──────────────────────────────────────────────
load_dotenv()

API_KEY = os.getenv("KAPT_API_KEY")

# ① 공동주택 단지 목록제공 서비스 (시도별)
APT_LIST_URL = (
    "http://apis.data.go.kr/1613000/AptListService3/getSidoAptList3"
)

# ② 공동주택 기본 정보제공 서비스
APT_DETAIL_URL = (
    "http://apis.data.go.kr/1613000/AptBasisInfoService1/getAphusBassInfo"
)

# 시도코드
SIDO_CODES = {
    "11": "서울특별시",
    "26": "부산광역시",
    "27": "대구광역시",
    "28": "인천광역시",
    "29": "광주광역시",
    "30": "대전광역시",
    "31": "울산광역시",
    "36": "세종특별자치시",
    "41": "경기도",
    "42": "강원특별자치도",
    "43": "충청북도",
    "44": "충청남도",
    "45": "전북특별자치도",
    "46": "전라남도",
    "47": "경상북도",
    "48": "경상남도",
    "50": "제주특별자치도",
}


def fetch_apt_list(
    sido_code: str = "11",
    num_of_rows: int = 20,
    page_no: int = 1,
) -> dict:
    """
    ① 시도코드로 해당 지역의 공동주택 단지 목록을 조회합니다.

    Args:
        sido_code: 시도코드 (예: 11 = 서울)
        num_of_rows: 한 페이지 결과 수
        page_no: 페이지 번호

    Returns:
        API 응답 JSON dict (단지코드 kaptCode 포함)
    """
    if not API_KEY:
        raise ValueError(
            "KAPT_API_KEY가 설정되지 않았습니다. "
            ".env 파일에 KAPT_API_KEY=... 를 추가하세요."
        )

    params = {
        "serviceKey": API_KEY,
        "sidoCode": sido_code,
        "numOfRows": num_of_rows,
        "pageNo": page_no,
        "resultType": "json",
    }

    sido_name = SIDO_CODES.get(sido_code, sido_code)
    print(f"[K-apt] 단지 목록 요청 중... 지역: {sido_name} (코드: {sido_code})")

    resp = requests.get(APT_LIST_URL, params=params, timeout=30)
    resp.raise_for_status()

    try:
        data = resp.json()
    except requests.exceptions.JSONDecodeError:
        print("[K-apt] JSON 파싱 실패. 원본 응답:")
        print(resp.text[:2000])
        return {"error": "JSON 파싱 실패", "raw": resp.text[:2000]}

    return data


def fetch_apt_detail(kapt_code: str) -> dict:
    """
    ② 단지코드(kaptCode)로 공동주택 기본 정보를 조회합니다.
    (세대수, 관리비, 주소, 준공일 등)

    ※ 이 API는 '공동주택 기본 정보제공 서비스'를 별도로 신청해야 사용 가능합니다.

    Args:
        kapt_code: 단지코드 (예: A10021295)

    Returns:
        API 응답 JSON dict
    """
    if not API_KEY:
        raise ValueError(
            "KAPT_API_KEY가 설정되지 않았습니다. "
            ".env 파일에 KAPT_API_KEY=... 를 추가하세요."
        )

    params = {
        "serviceKey": API_KEY,
        "kaptCode": kapt_code,
    }

    print(f"[K-apt] 기본 정보 요청 중... 단지코드: {kapt_code}")

    resp = requests.get(APT_DETAIL_URL, params=params, timeout=30)

    if resp.status_code == 404:
        return {
            "error": "기본 정보 API 미신청 또는 엔드포인트 변경",
            "message": (
                "'국토교통부_공동주택 기본 정보제공 서비스'를 "
                "공공데이터포털에서 별도로 활용 신청해야 합니다. "
                "https://www.data.go.kr/data/15058453/openapi.do"
            ),
        }

    if resp.status_code == 500 or "Unexpected errors" in resp.text:
        return {
            "error": "서버 내부 오류 (데이터 미동기화)",
            "message": (
                "공공데이터포털 서버에서 오류(Unexpected errors)가 발생했습니다.\n"
                "API를 방금 신청하셨다면, 국토교통부 서버와 공공데이터포털 간의 "
                "동기화에 1시간에서 최대 24시간이 소요될 수 있습니다.\n"
                "동기화가 완료된 후 다시 시도해 주세요."
            ),
            "raw": resp.text[:200]
        }

    resp.raise_for_status()

    try:
        data = resp.json()
    except requests.exceptions.JSONDecodeError:
        # XML 응답 처리
        print("[K-apt] XML 응답:")
        print(resp.text[:2000])
        return {"raw_xml": resp.text[:2000]}

    return data


def main():
    """서울 단지 목록 조회 → 첫 번째 단지 기본정보 조회 (테스트)"""
    try:
        # ── Step 1: 서울 단지 목록 조회 ──
        print("=" * 50)
        print("📋 Step 1: 공동주택 단지 목록 조회 (서울)")
        print("=" * 50)
        list_data = fetch_apt_list(sido_code="11", num_of_rows=5)
        print(json.dumps(list_data, ensure_ascii=False, indent=2))

        # ── Step 2: 첫 번째 단지의 기본정보 조회 ──
        kapt_code = None
        try:
            items = (
                list_data.get("response", {})
                .get("body", {})
                .get("items", [])
            )
            if isinstance(items, list) and len(items) > 0:
                kapt_code = items[0].get("kaptCode")
                kapt_name = items[0].get("kaptName", "")
            elif isinstance(items, dict):
                kapt_code = items.get("kaptCode")
                kapt_name = items.get("kaptName", "")
        except (AttributeError, KeyError, IndexError):
            pass

        if kapt_code:
            print("\n" + "=" * 50)
            print(f"🏢 Step 2: 기본 정보 조회 ({kapt_name}, 코드: {kapt_code})")
            print("=" * 50)
            detail_data = fetch_apt_detail(kapt_code)
            print(json.dumps(detail_data, ensure_ascii=False, indent=2))
        else:
            print("\n⚠️ 단지 목록에서 kaptCode를 찾지 못했습니다.")

    except ValueError as e:
        print(f"❌ 오류: {e}", file=sys.stderr)
        sys.exit(1)
    except requests.RequestException as e:
        print(f"❌ API 요청 실패: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n✅ K-apt API 호출 완료!")


if __name__ == "__main__":
    main()
