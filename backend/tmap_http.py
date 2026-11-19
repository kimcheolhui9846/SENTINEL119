"""Stdlib HTTPS transport for the TMAP route API."""

import http.client
import json
import math
import re

_HOST = "apis.openapi.sk.com"
_PATH = "/tmap/routes?version=1"
_TIMEOUT = 3
_MAX_BYTES = 2 * 1024 * 1024
_MAX_JSON_DEPTH = 64
_KEY_RE = re.compile(r"[\x21-\x7e]+")


class RouteFetchError(ValueError):
    """Route fetch failure; only a short reason code is exposed."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _reject_constant(_name):
    raise ValueError("invalid JSON")


def _parse_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("invalid JSON")
    return value


def _check_json_depth(text):
    """Count containers outside strings before passing input to the decoder."""
    depth = 0
    in_string = escaped = False
    for char in text:
        if escaped:
            escaped = False
        elif in_string:
            if char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            if depth > _MAX_JSON_DEPTH:
                raise ValueError("invalid JSON")
        elif char in "]}":
            depth -= 1


def strict_json_loads(data):
    """Parse UTF-8 (BOM tolerated) JSON bytes strictly; raise generic ValueError."""
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("invalid JSON")
    try:
        text = data.decode("utf-8-sig")
        _check_json_depth(text)
        return json.loads(
            text,
            parse_float=_parse_float,
            parse_constant=_reject_constant,
        )
    except (ValueError, RecursionError):
        pass
    raise ValueError("invalid JSON")


def _valid_key(key):
    return isinstance(key, str) and _KEY_RE.fullmatch(key) is not None


def _build_body(start, end):
    try:
        start_x, start_y = (float(v) for v in start)
        end_x, end_y = (float(v) for v in end)
        if not all(math.isfinite(v) for v in (start_x, start_y, end_x, end_y)):
            raise ValueError
        return json.dumps(
            {
                "startX": start_x,
                "startY": start_y,
                "endX": end_x,
                "endY": end_y,
                "reqCoordType": "WGS84GEO",
                "resCoordType": "WGS84GEO",
                "searchOption": 0,
                "totalValue": 1,
            },
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        pass
    raise RouteFetchError("invalid_request")


def _post(key, body):
    headers = {
        "appKey": key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    conn = None
    try:
        conn = http.client.HTTPSConnection(_HOST, timeout=_TIMEOUT)
        conn.request("POST", _PATH, body=body, headers=headers)
        response = conn.getresponse()
        if not 200 <= response.status < 300:
            raise RouteFetchError("http_error")
        data = response.read(_MAX_BYTES + 1)
        if not isinstance(data, (bytes, bytearray)) or len(data) > _MAX_BYTES:
            raise RouteFetchError("malformed_response")
        return bytes(data)
    except TimeoutError:
        reason = "timeout"
    except (OSError, http.client.HTTPException):
        reason = "network_error"
    finally:
        if conn is not None:
            try:
                conn.close()
            except (OSError, http.client.HTTPException):
                pass
    raise RouteFetchError(reason)


def make_tmap_fetcher(key):
    """Return fetch(start, end) -> parsed TMAP JSON; (lon, lat) tuples."""

    def fetch(start, end):
        if not _valid_key(key):
            raise RouteFetchError("offline")
        raw = _post(key, _build_body(start, end))
        try:
            return strict_json_loads(raw)
        except ValueError:
            pass
        raise RouteFetchError("malformed_response")

    return fetch
