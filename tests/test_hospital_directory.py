"""HTTP contract for the NMC directory, using an in-process upstream."""

import os
import unittest
from datetime import datetime, timezone
from unittest.mock import patch
from xml.etree.ElementTree import Element, SubElement, tostring

import httpx
from fastapi.testclient import TestClient

from backend.api import app, create_app


def hospital(hpid, *, lat="37.566", lon="127.042", category="의원", emergency=""):
    return {
        "hpid": hpid,
        "dutyName": "테스트 의료기관",
        "dutyAddr": "서울특별시 성동구 테스트로 1",
        "dutyDivNam": category,
        "dutyEmcls": emergency,
        "wgs84Lat": lat,
        "wgs84Lon": lon,
    }


def xml_page(total, *rows, count=100, page=1, code="00"):
    response = Element("response")
    header = SubElement(response, "header")
    SubElement(header, "resultCode").text = code
    body = SubElement(response, "body")
    SubElement(body, "numOfRows").text = str(count)
    SubElement(body, "pageNo").text = str(page)
    SubElement(body, "totalCount").text = str(total)
    items = SubElement(body, "items")
    for values in rows:
        item = SubElement(items, "item")
        for field, value in values.items():
            SubElement(item, field).text = value
    return httpx.Response(200, content=tostring(response, encoding="utf-8"))


class HospitalDirectoryTests(unittest.TestCase):
    path = "/api/hospital-locations"

    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"NMC_HOSPITAL_API_KEY": "test%2Bkey%2F%3D"}))
        self.requests = []
        self.responses = []
        self.now = 100.0

    def connect(self, *, ttl_clock=False):
        if ttl_clock:
            self.enterContext(patch("backend.hospital_directory._monotonic", lambda: self.now))
        self.enterContext(patch("backend.hospital_directory.httpx.AsyncHTTPTransport",
                                return_value=httpx.MockTransport(self.upstream)))
        return self.enterContext(TestClient(create_app()))

    def upstream(self, request):
        self.requests.append(request)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def test_module_app_without_key_reports_unconfigured(self):
        del os.environ["NMC_HOSPITAL_API_KEY"]
        with TestClient(app) as client:
            response = client.get(self.path)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "district": "성동구", "source": "nmc", "status": "unconfigured",
            "fetched_at": None, "total_count": 0, "items": [], "reason": "missing_api_key",
        })

    def test_unknown_district_or_url_is_rejected(self):
        with TestClient(create_app()) as client:
            for district in ("수원시", "https://example.invalid/", "", "성동구 "):
                with self.subTest(district=district):
                    response = client.get(self.path, params={"district": district})
                    self.assertEqual(response.status_code, 422)
                    self.assertEqual(response.json(), {"detail": "invalid_district"})

    def test_full_district_list_accepts_all_25(self):
        del os.environ["NMC_HOSPITAL_API_KEY"]
        names = (
            "강남구 강동구 강북구 강서구 관악구 광진구 구로구 금천구 노원구 도봉구 "
            "동대문구 동작구 마포구 서대문구 서초구 성동구 성북구 송파구 양천구 "
            "영등포구 용산구 은평구 종로구 중구 중랑구"
        )
        with TestClient(create_app()) as client:
            for district in names.split():
                with self.subTest(district=district):
                    response = client.get(self.path, params={"district": district})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["district"], district)

    def test_paginates_deduplicates_and_keeps_every_valid_category(self):
        client = self.connect()
        self.responses.extend([
            xml_page(5, hospital("H1"), hospital("H2", category="종합병원", emergency="G201"),
                     hospital("H3", lat="127.04", lon="37.56"), count=3),
            xml_page(5, hospital("H2"), hospital("", lat="NaN"), count=3, page=2),
        ])
        result = client.get(self.path, params={"district": "중구"}).json()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["source"], "nmc")
        self.assertEqual(result["district"], "중구")
        self.assertIsNone(result["reason"])
        self.assertEqual(result["total_count"], 5)
        fetched_at = datetime.fromisoformat(result["fetched_at"])
        self.assertEqual(fetched_at.utcoffset(), timezone.utc.utcoffset(None))
        self.assertEqual(result["items"], [
            {"id": "H1", "name": "테스트 의료기관", "address": "서울특별시 성동구 테스트로 1",
             "lat": 37.566, "lon": 127.042, "category": "의원", "emergency_code": ""},
            {"id": "H2", "name": "테스트 의료기관", "address": "서울특별시 성동구 테스트로 1",
             "lat": 37.566, "lon": 127.042, "category": "종합병원", "emergency_code": "G201"},
        ])
        self.assertEqual([request.url.params["pageNo"] for request in self.requests], ["1", "2"])
        for request in self.requests:
            self.assertEqual(str(request.url.copy_with(query=None)),
                "https://apis.data.go.kr/B552657/HsptlAsembySearchService/getHsptlMdcncListInfoInqire")
            self.assertEqual(request.url.params["Q0"], "서울특별시")
            self.assertEqual(request.url.params["Q1"], "중구")
            self.assertEqual(request.url.params["serviceKey"], "test+key/=")
            self.assertEqual(request.url.params["numOfRows"], "100")

    def test_raw_key_and_legitimate_zero_result(self):
        os.environ["NMC_HOSPITAL_API_KEY"] = "raw+key/="
        client = self.connect()
        self.responses.append(xml_page(0))
        result = client.get(self.path).json()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["total_count"], 0)
        self.assertEqual(result["items"], [])
        self.assertIsNotNone(result["fetched_at"])
        self.assertEqual(self.requests[0].url.params["serviceKey"], "raw+key/=")

    def test_missing_nonfinite_and_out_of_range_coordinates_are_skipped(self):
        client = self.connect()
        self.responses.append(xml_page(
            7, hospital("", lat="37.5"), hospital("blank", lon=""),
            hospital("text", lat="unknown"), hospital("nan", lat="NaN"),
            hospital("infinity", lon="Infinity"), hospital("far", lon="181"),
            hospital("origin", lat="0", lon="0"),
        ))
        result = client.get(self.path).json()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["total_count"], 7)
        self.assertEqual(result["items"], [])

    def test_cached_result_is_per_district_and_stale_uses_original_timestamp(self):
        client = self.connect(ttl_clock=True)
        self.responses.extend([
            xml_page(1, hospital("H1")), xml_page(1, hospital("H2")),
            httpx.ReadTimeout("private raw response"),
        ])
        first = client.get(self.path).json()
        self.assertEqual(client.get(self.path).json(), first)
        other = client.get(self.path, params={"district": "중구"}).json()
        self.assertEqual(other["items"][0]["id"], "H2")
        self.now += 3600
        stale = client.get(self.path).json()
        self.assertEqual(stale, {**first, "status": "stale", "reason": "upstream_timeout"})
        self.assertEqual(len(self.requests), 3)

    def test_incomplete_pagination_never_reports_partial_success(self):
        client = self.connect()
        for second_page in (
            xml_page(3, count=2, page=2),
            xml_page(4, hospital("H1"), hospital("H2"), count=2, page=2),
            xml_page(3, hospital("H3"), count=2, page=1),
        ):
            with self.subTest(body=second_page.content):
                total = 4 if b"<totalCount>4" in second_page.content else 3
                self.responses.extend([xml_page(total, hospital("H1"), hospital("H2"), count=2),
                                       second_page])
                result = client.get(self.path).json()
                self.assertEqual(result["status"], "unavailable")
                self.assertEqual(result["reason"], "incomplete_response")
                self.assertEqual(result["items"], [])
                self.assertIsNone(result["fetched_at"])

    def test_upstream_errors_are_sanitized_and_redirects_are_not_followed(self):
        client = self.connect()
        for response, reason in (
            (httpx.Response(307, headers={"Location": "https://example.invalid/redirect"}), "upstream_unavailable"),
            (xml_page(0, code="30"), "upstream_unavailable"),
            (httpx.Response(200, content=b"private invalid format"), "invalid_response"),
            (httpx.Response(200, content=b"x" * 1_048_577), "response_too_large"),
            (httpx.ConnectError("private transport exception"), "upstream_unavailable"),
            (xml_page(3000, hospital("H1"), count=1), "incomplete_response"),
        ):
            with self.subTest(reason=reason):
                self.responses.append(response)
                start = len(self.requests)
                result = client.get(self.path).json()
                self.assertEqual(result, {"district": "성동구", "source": "nmc",
                    "status": "unavailable", "reason": reason, "fetched_at": None,
                    "total_count": 0, "items": []})
                self.assertEqual(len(self.requests), start + 1)
