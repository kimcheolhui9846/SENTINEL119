"""Approximate incident area for call files: the AI Hub ``address`` (시 구 법정동) on Seoul dong boundaries.

119 receives a caller's approximate position from the carrier; the AI Hub record keeps it only at the
법정동 level. The incident is drawn as that dong and its centre, never as an exact address. Vehicles and
hospitals around it are virtual and placed deterministically from the dong code, so a replay is stable.
"""

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path


DONGS_PATH = Path(__file__).with_name("data") / "seoul-dongs.geojson"
# Urban response speed with a road detour factor: minutes = km * 1.3 / 25 km/h * 60.
_KM_PER_DEG_LAT = 111.0
_ROAD_FACTOR, _SPEED_KMH = 1.3, 25.0


@lru_cache(maxsize=1)
def _dongs():
    data = json.loads(DONGS_PATH.read_text(encoding="utf-8"))
    return {(f["properties"]["district"], f["properties"]["name"]): f for f in data["features"]}


def _inside(lon, lat, ring):
    hit = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def _random_point(geometry, seed):
    """A reproducible point inside the dong (rejection sampling over its bounding box)."""
    import random
    polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    rings = [poly[0] for poly in polygons]
    xs = [x for ring in rings for x, _ in ring]
    ys = [y for ring in rings for _, y in ring]
    rng = random.Random(seed)
    for _ in range(2000):
        lon, lat = rng.uniform(min(xs), max(xs)), rng.uniform(min(ys), max(ys))
        if any(_inside(lon, lat, ring) for ring in rings):
            return {"lon": round(lon, 6), "lat": round(lat, 6)}
    return None


def find_area(address, seed=None):
    """'서울특별시 성동구 성수동1가' -> area dict, or None outside the Seoul dong map.

    With ``seed`` the incident point is a reproducible random point inside the dong (demo stand-in for
    the caller position); without it, the dong centre."""
    parts = (address or "").split()
    if len(parts) != 3 or not parts[0].startswith("서울"):
        return None
    feature = _dongs().get((parts[1], parts[2]))
    if feature is None:
        return None
    props = feature["properties"]
    dong_center = {"lon": props["center"][0], "lat": props["center"][1]}
    point = _random_point(feature["geometry"], seed) if seed is not None else None
    return {"city": parts[0], "district": props["district"], "dong": props["name"], "code": props["code"],
            "center": point or dong_center, "dong_center": dong_center, "random_point": point is not None,
            "geometry": feature["geometry"], "precision": "법정동",
            "source": "AI Hub 71768 address · JUSO 2015 법정동 경계 (southkorea/seoul-maps, Apache-2.0)"}


def _offset(center, bearing_deg, km):
    lat = center["lat"] + km * math.cos(math.radians(bearing_deg)) / _KM_PER_DEG_LAT
    lon = center["lon"] + km * math.sin(math.radians(bearing_deg)) / (
        _KM_PER_DEG_LAT * math.cos(math.radians(center["lat"])))
    return {"lat": round(lat, 6), "lon": round(lon, 6)}


def _eta(km):
    return max(1, round(km * _ROAD_FACTOR / _SPEED_KMH * 60))


def _layout(code, count, near, far):
    seed = int(hashlib.sha256(code.encode()).hexdigest()[:8], 16)
    for i in range(count):
        bearing = (seed % 360 + i * 360 / count + (seed >> (i + 3)) % 40) % 360
        km = near + (far - near) * (((seed >> (2 * i + 5)) % 100) / 100)
        yield bearing, round(km, 2)


def virtual_resources(area):
    """Two ambulances, a pump and a rescue unit within a few km, and three hospitals a little further."""
    center = area["center"]
    units = []
    kinds = [("A", "구급차 A", "ambulance", "기본 구급 장비"), ("B", "구급차 B", "ambulance", "AED · 산소 · 1급 응급구조사"),
             ("F1", "펌프차 F1", "fire", "펌프 · 소방호스"), ("R1", "구조공작차 R1", "rescue", "구조 장비")]
    for (key, name, kind, detail), (bearing, km) in zip(kinds, _layout(area["code"], len(kinds), 0.8, 3.2)):
        units.append({"id": key, "name": name, "kind": kind, "eta_min": _eta(km), "distance_km": km,
                      "detail": f"{detail} · 가상 차량", **_offset(center, bearing, km)})
    hospitals = []
    for i, (bearing, km) in enumerate(_layout(area["code"][::-1], 3, 1.5, 5.0), start=1):
        hospitals.append({"id": f"H{i}", "name": f"{area['district']} 가상 병원 {i}", "eta_min": _eta(km),
                          "distance_km": km, "detail": "수용 정보 시연용 가상 값", **_offset(center, bearing, km)})
    return units, hospitals
