"""Public NMC positions with a separate, versioned exercise overlay."""

import asyncio
from datetime import datetime
import json
import math
from pathlib import Path
import re

from .hospital_directory import HospitalDirectory


FACILITIES_PATH = Path(__file__).with_name("data") / "drill-facilities.json"
_PUBLIC_FIELDS = ("id", "name", "address", "lat", "lon", "category", "emergency_code")
_FACILITY_TIMEOUT = 8
_LIMIT = 3


def _emergency(facility):
    return re.fullmatch(r"G00[1-9]", facility["emergency_code"].strip(), re.IGNORECASE) is not None


def _candidate_rows(payload, district):
    if not isinstance(payload, dict) or payload.get("district") != district:
        return None, []
    date = payload.get("fetched_at")
    items = payload.get("items")
    if not isinstance(date, str) or not isinstance(items, list):
        return None, []
    try:
        if datetime.fromisoformat(date).utcoffset() is None:
            return None, []
    except ValueError:
        return None, []

    candidates = {}
    for candidate in items:
        if not isinstance(candidate, dict):
            continue
        if not all(isinstance(candidate.get(key), str)
                   for key in ("id", "name", "address", "category", "emergency_code")):
            continue
        if not (candidate["id"].strip() and candidate["name"].strip() and candidate["address"].strip()
                and candidate["category"].strip().endswith("병원")):
            continue
        lat, lon = candidate.get("lat"), candidate.get("lon")
        if (type(lat) not in (int, float) or type(lon) not in (int, float)
                or not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat, lon) == (0, 0)):
            continue
        candidates.setdefault(candidate["id"], {key: candidate[key] for key in _PUBLIC_FIELDS})
    return date, list(candidates.values())


def _exercise_hospitals(rows, caller):
    lon_scale = math.cos(math.radians(caller["lat"]))

    def distance(facility):
        return ((facility["lat"] - caller["lat"]) ** 2
                + ((facility["lon"] - caller["lon"]) * lon_scale) ** 2)

    ordered = sorted(rows, key=lambda facility: (distance(facility), facility["id"]))
    emergency = next((facility for facility in ordered if _emergency(facility)), None)
    selected = ordered[:_LIMIT]
    if emergency is not None and emergency not in selected:
        selected[-1] = emergency
    return [
        {**facility, "location_source": "nmc", "operational_source": "simulation",
         "operational_profile": "drill-v1",
         "capabilities": ["cardiac_arrest"] if facility is emergency else [],
         "available": facility is emergency}
        for facility in selected
    ]


class DrillFacilities:
    def __init__(self, directory=None, snapshot_path=FACILITIES_PATH):
        self._directory = directory if directory is not None else HospitalDirectory()
        self._snapshot_path = snapshot_path

    async def for_drill(self, mode, district, caller):
        if mode == "live":
            try:
                async with asyncio.timeout(_FACILITY_TIMEOUT):
                    live = await self._directory.locations(district)
            except TimeoutError:
                live = None
            if isinstance(live, dict) and live.get("source") == "nmc" and live.get("status") == "ok":
                fetched_at, rows = _candidate_rows(live, district)
                if rows:
                    return _exercise_hospitals(rows, caller), {
                        "provider": "nmc", "status": "live", "fetched_at": fetched_at,
                    }

        try:
            snapshot = json.loads(self._snapshot_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            snapshot = None
        fetched_at, rows = _candidate_rows(snapshot, district)
        if rows:
            return _exercise_hospitals(rows, caller), {
                "provider": "nmc", "status": "snapshot", "fetched_at": fetched_at,
            }
        return [], {"provider": "nmc", "status": "unavailable", "fetched_at": None}
