"""District-level medical facility locations from the NMC public directory."""

import asyncio
import os
from datetime import datetime, timezone
from time import monotonic as _monotonic
from urllib.parse import unquote
from xml.etree import ElementTree

import httpx


DEFAULT_DISTRICT = "성동구"
SEOUL_DISTRICTS = frozenset((
    "강남구 강동구 강북구 강서구 관악구 광진구 구로구 금천구 노원구 도봉구 "
    "동대문구 동작구 마포구 서대문구 서초구 성동구 성북구 송파구 양천구 "
    "영등포구 용산구 은평구 종로구 중구 중랑구"
).split())
_ENDPOINT = "https://apis.data.go.kr/B552657/HsptlAsembySearchService/getHsptlMdcncListInfoInqire"
_PAGE_SIZE = 100
_MAX_PAGES = 25
_MAX_PAGE_BYTES = 1_048_576
_TIMEOUT_SECONDS = 10
_TTL_SECONDS = 300


class _UpstreamError(Exception):
    """A credential-free reason for a failed directory refresh."""


def _value(element, field):
    return (element.findtext(field) or "").strip()


def _facility(row):
    facility_id = _value(row, "hpid")
    if not facility_id:
        return None
    try:
        lat = float(_value(row, "wgs84Lat"))
        lon = float(_value(row, "wgs84Lon"))
    except ValueError:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat, lon) == (0, 0):
        return None
    return {
        "id": facility_id,
        "name": _value(row, "dutyName"),
        "address": _value(row, "dutyAddr"),
        "lat": lat,
        "lon": lon,
        "category": _value(row, "dutyDivNam") or _value(row, "dutyDivName") or _value(row, "dutyDiv"),
        "emergency_code": _value(row, "dutyEmcls"),
    }


async def _read_page(transport, district, key, page):
    request = httpx.Request("GET", _ENDPOINT, params={
        "serviceKey": key, "Q0": "서울특별시", "Q1": district,
        "pageNo": page, "numOfRows": _PAGE_SIZE,
    }, headers={"Accept-Encoding": "identity"})
    # The transport keeps the credential-bearing URL out of HTTPX client logs.
    response = await transport.handle_async_request(request)
    try:
        if not response.is_success:
            raise _UpstreamError("upstream_unavailable")
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise _UpstreamError("invalid_response")
        content = bytearray()
        async for chunk in response.aiter_bytes():
            if len(content) + len(chunk) > _MAX_PAGE_BYTES:
                raise _UpstreamError("response_too_large")
            content.extend(chunk)
    finally:
        await response.aclose()

    try:
        xml = content.decode("utf-8-sig")
        if "<!DOCTYPE" in xml or "<!ENTITY" in xml:
            raise _UpstreamError("invalid_response")
        root = ElementTree.fromstring(xml)
        if _value(root, "header/resultCode") != "00":
            raise _UpstreamError("upstream_unavailable")
        total = int(_value(root, "body/totalCount"))
        size = int(_value(root, "body/numOfRows") or _PAGE_SIZE)
        reported_page = int(_value(root, "body/pageNo") or page)
    except (ValueError, ElementTree.ParseError):
        raise _UpstreamError("invalid_response") from None
    if total < 0 or (total > 0 and not 1 <= size <= _PAGE_SIZE):
        raise _UpstreamError("invalid_response")
    if reported_page != page:
        raise _UpstreamError("incomplete_response")
    return total, size, root.findall("body/items/item")


async def _fetch_directory(district, key):
    facilities = {}
    page_keys = set()
    previous_total = previous_size = None
    seen = 0
    async with asyncio.timeout(_TIMEOUT_SECONDS), httpx.AsyncHTTPTransport(trust_env=False) as transport:
        for page in range(1, _MAX_PAGES + 1):
            total, size, rows = await _read_page(transport, district, key, page)
            if (previous_total is not None and (total, size) != (previous_total, previous_size)
                    or total > size * _MAX_PAGES or len(rows) != min(size, total - seen)):
                raise _UpstreamError("incomplete_response")
            previous_total, previous_size = total, size
            # Keep duplicate ids across different pages, but reject a replayed page.
            page_key = tuple(sorted(
                _value(row, "hpid") or ElementTree.tostring(row, encoding="unicode")
                for row in rows
            ))
            if page_key in page_keys:
                raise _UpstreamError("incomplete_response")
            page_keys.add(page_key)
            for row in rows:
                facility = _facility(row)
                if facility is not None:
                    facilities.setdefault(facility["id"], facility)
            seen += len(rows)
            if seen == total:
                return total, list(facilities.values())
    raise _UpstreamError("incomplete_response")


class HospitalDirectory:
    def __init__(self):
        self._cache = {}
        self._locks = {district: asyncio.Lock() for district in SEOUL_DISTRICTS}

    async def locations(self, district):
        result = {
            "district": district, "source": "nmc", "status": "unconfigured",
            "fetched_at": None, "total_count": 0, "items": [], "reason": "missing_api_key",
        }
        key = os.environ.get("NMC_HOSPITAL_API_KEY", "").strip()
        if not key:
            return result

        async with self._locks[district]:
            cached = self._cache.get(district)
            if cached and _monotonic() < cached[0]:
                return cached[1]
            try:
                total, items = await _fetch_directory(district, unquote(key))
            except (TimeoutError, httpx.TimeoutException):
                reason = "upstream_timeout"
            except httpx.HTTPError:
                reason = "upstream_unavailable"
            except _UpstreamError as error:
                reason = str(error)
            else:
                result.update(
                    status="ok", reason=None, total_count=total, items=items,
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                )
                self._cache[district] = (_monotonic() + _TTL_SECONDS, result)
                return result
            if cached:
                return {**cached[1], "status": "stale", "reason": reason}
            return {**result, "status": "unavailable", "reason": reason}
