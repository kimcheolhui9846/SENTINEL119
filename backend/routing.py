"""Road routes for the dispatch map: TMAP car routing when TMAP_APP_KEY is set, else the public OSRM
demo server (OpenStreetMap roads), else a straight line with a distance-based ETA.

Adapted from ermct-backend app/distance_logic.py (TMAP request and response handling). Results are cached
by rounded endpoints so repeated demo runs do not spend the TMAP quota, and concurrent calls are capped.
"""

import asyncio
import math
import os

from pathlib import Path

import httpx


_TMAP_URL = "https://apis.openapi.sk.com/tmap/routes?version=1&format=json"
_OSRM_URL = "https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
_TIMEOUT = 8.0
_CACHE: dict = {}
_LIMIT = asyncio.Semaphore(4)
# Straight-line fallback: urban speed with a detour factor.
_ROAD_FACTOR, _SPEED_KMH = 1.3, 25.0


_ENV_FILE = Path(__file__).resolve().parents[1] / ".env.local"  # git-ignored; KEY=value lines


def _setting(name):
    """Environment first, then the repository's git-ignored .env.local (never logged or returned)."""
    value = os.getenv(name, "").strip()
    if value or not _ENV_FILE.is_file():
        return value
    for line in _ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
        key, sep, raw = line.partition("=")
        if sep and key.strip() == name:
            return raw.strip().strip('"').strip("'")
    return ""


def provider():
    return "tmap" if _setting("TMAP_APP_KEY") else "osrm"


def _haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def _line(start, end):
    km = _haversine_km(start, end) * _ROAD_FACTOR
    return {"path": [start, end], "distance_m": round(km * 1000), "duration_s": round(km / _SPEED_KMH * 3600),
            "source": "straight_line"}


async def _tmap(client, key, start, end):
    body = {"startX": str(start["lon"]), "startY": str(start["lat"]), "endX": str(end["lon"]), "endY": str(end["lat"]),
            "reqCoordType": "WGS84GEO", "resCoordType": "WGS84GEO", "searchOption": "0"}
    response = await client.post(_TMAP_URL, json=body, headers={"appKey": key})
    response.raise_for_status()
    features = response.json().get("features") or []
    path, distance, duration = [], None, None
    for feature in features:
        props, geometry = feature.get("properties") or {}, feature.get("geometry") or {}
        if distance is None and "totalDistance" in props:
            distance, duration = props["totalDistance"], props.get("totalTime")
        lines = ([geometry.get("coordinates")] if geometry.get("type") == "LineString"
                 else geometry.get("coordinates") if geometry.get("type") == "MultiLineString" else [])
        for line in lines or []:
            for lon, lat in line:
                point = {"lat": float(lat), "lon": float(lon)}
                if not path or path[-1] != point:
                    path.append(point)
    if len(path) < 2 or distance is None:
        raise ValueError("TMAP route without a path")
    return {"path": path, "distance_m": int(distance), "duration_s": int(duration or 0), "source": "tmap"}


async def _osrm(client, start, end):
    url = _OSRM_URL.format(lon1=start["lon"], lat1=start["lat"], lon2=end["lon"], lat2=end["lat"])
    response = await client.get(url, params={"overview": "full", "geometries": "geojson"})
    response.raise_for_status()
    routes = response.json().get("routes") or []
    if not routes:
        raise ValueError("OSRM returned no route")
    coordinates = routes[0]["geometry"]["coordinates"]
    return {"path": [{"lat": lat, "lon": lon} for lon, lat in coordinates],
            "distance_m": round(routes[0]["distance"]), "duration_s": round(routes[0]["duration"]), "source": "osrm"}


async def route(start, end):
    """{path: [{lat, lon}], distance_m, duration_s, source}; never raises."""
    key = (round(start["lat"], 5), round(start["lon"], 5), round(end["lat"], 5), round(end["lon"], 5))
    if key in _CACHE:
        return _CACHE[key]
    tmap_key = _setting("TMAP_APP_KEY")
    result = None
    async with _LIMIT:
        async with httpx.AsyncClient(timeout=_TIMEOUT, trust_env=False,
                                     headers={"User-Agent": "sentinel119-demo"}) as client:
            for attempt in ((lambda: _tmap(client, tmap_key, start, end)) if tmap_key else None,
                            lambda: _osrm(client, start, end)):
                if attempt is None:
                    continue
                try:
                    result = await attempt()
                    break
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    continue
    result = result or _line(start, end)
    _CACHE[key] = result
    return result
