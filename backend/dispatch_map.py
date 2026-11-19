"""Dispatch map: which units a committed call type sends, from which station, along which road.

Units wait at real Seoul fire stations (OpenStreetMap amenity=fire_station): every 119안전센터 holds an
ambulance and a pump engine, every 소방서 a rescue unit (구조대) and a pump engine. When SENTINEL_ZORO
commits the dispatch type the plan below picks, per unit kind, the station with the shortest road ETA.

EMS rules quote the knowledge base (「119구급대원 현장응급처치 표준지침」 2023, 수보단계 중증응급환자 인지
프로토콜 p.156-160, 구급활동의 기록). The KB has no fire or rescue dispatch composition, so those plans are
the demo default and are labelled as such.
"""

import asyncio
from functools import lru_cache
import json
import math
import zlib
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from .routing import provider, route


STATIONS_PATH = Path(__file__).with_name("data") / "seoul-fire-stations.json"
_CANDIDATES_PER_KIND = 4

UNIT_NAMES = {"ambulance": "구급차", "pump": "펌프차", "rescue": "구조공작차"}
_KB = "119구급대원 현장응급처치 표준지침(2023)"
_RULES = {
    "구급": [{"kind": "ambulance", "role": "일반구급대",
              "reason": "구급 신고: 가장 가까운 구급대 출동",
              "source": f"{_KB} · 수보단계 중증응급환자 인지 프로토콜 (p.156)"}],
    "화재": [{"kind": "pump", "role": "화재진압대", "reason": "화재 신고: 가장 가까운 펌프차 출동",
              "source": "시연용 기본 편성 (지식베이스에 화재 출동 편성 없음)"},
             {"kind": "pump", "role": "화재진압대 (2차)", "reason": "화재 신고: 두 번째로 가까운 펌프차 동시 출동",
              "source": "시연용 기본 편성 (지식베이스에 화재 출동 편성 없음)"},
             {"kind": "ambulance", "role": "구급대 대기", "reason": "화재 현장 인명피해 대비 구급차 동행",
              "source": "시연용 기본 편성 (지식베이스에 화재 출동 편성 없음)"}],
    "구조": [{"kind": "rescue", "role": "구조대", "reason": "구조 신고: 가장 가까운 구조대 출동",
              "source": "시연용 기본 편성 (지식베이스에 구조 출동 편성 없음)"},
             {"kind": "ambulance", "role": "구급대", "reason": "구조 대상자 처치·이송을 위한 구급차 동시 출동",
              "source": f"{_KB} · 구급활동의 기록 ('구조대가 동시에 출동한 사건')"}],
    "기타": [],
}
# Upgrades once subtype / symptom / urgency are committed (EMS only; quoted from the KB).
_EARLY = {"심정지", "심정지·호흡정지"}
_SPECIAL = {"임산부": "응급분만", "부상": "중증외상", "사고": "중증외상",
            "흉통·가슴불편감·심계항진": "심인성 흉통", "마비": "뇌졸중"}


def plan(dispatch, subtype=None, symptom=None, urgency=None):
    """Ordered unit requests [{kind, role, reason, source}] for the committed decisions."""
    if dispatch not in _RULES:
        raise ValueError("unknown dispatch type")
    steps = [dict(step) for step in _RULES[dispatch]]
    if dispatch != "구급":
        return steps
    if subtype in _EARLY or symptom in _EARLY:
        steps[0].update(role="조기 긴급출동", reason="심정지(의심): 구급차 조기 긴급출동 후 전화 CPR 지도",
                        source=f"{_KB} · 수보단계 중증응급환자 인지 프로토콜, 조기 긴급출동 기준 (p.157)")
        steps.append({"kind": "pump", "role": "펌뷸런스 (다중출동 CPR)",
                      "reason": "심정지: 펌뷸런스 다중출동으로 CPR 인력 지원",
                      "source": f"{_KB} · 구급활동의 기록, 다중출동 CPR(구급차/펌뷸런스)"})
    else:
        special = _SPECIAL.get(subtype) or _SPECIAL.get(symptom)
        if special and urgency == "상":
            steps[0].update(role=f"특별구급대 ({special})", reason=f"{special} 의심 · 긴급도 상: 특별구급대 긴급출동",
                            source=f"{_KB} · 수보단계 중증응급환자 인지 프로토콜, 특별구급대 출동 기준 (p.157)")
    return steps


@lru_cache(maxsize=1)
def stations():
    data = json.loads(STATIONS_PATH.read_text(encoding="utf-8"))
    return data["stations"], data["attribution"]


def fleet():
    units = []
    for s in stations()[0]:
        kinds = ("ambulance", "pump") if s["kind"] == "center" else ("rescue", "pump")
        for kind in kinds:
            units.append({"id": f"{s['id']}-{kind}", "name": f"{s['name']} {UNIT_NAMES[kind]}", "kind": kind,
                          "station": s["name"], "district": s["district"], "lat": s["lat"], "lon": s["lon"]})
    return units


def _km(a, b):
    dy = (a["lat"] - b["lat"]) * 111.0
    dx = (a["lon"] - b["lon"]) * 111.0 * math.cos(math.radians(a["lat"]))
    return math.hypot(dx, dy)


async def assign(incident, steps, busy=(), positions=None):
    """Pick, per requested unit, the free unit of that kind with the shortest road ETA.

    ``positions`` maps unit IDs to where a patrolling unit is now; its route starts there."""
    taken, out = set(busy), []
    units = [{**u, **(positions or {}).get(u["id"], {})} for u in fleet()]
    for step in steps:
        pool = sorted((u for u in units if u["kind"] == step["kind"] and u["id"] not in taken),
                      key=lambda u: _km(u, incident))[:_CANDIDATES_PER_KIND]
        if not pool:
            out.append({**step, "unit": None})
            continue
        routes = await asyncio.gather(*(route(u, incident) for u in pool))
        unit, path = min(zip(pool, routes), key=lambda pair: pair[1]["duration_s"])
        taken.add(unit["id"])
        out.append({**step, "unit": unit, "route": path})
    return out


class DispatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lat: Annotated[float, Field(ge=37.0, le=38.0)]
    lon: Annotated[float, Field(ge=126.5, le=127.5)]
    dispatch: str
    subtype: str | None = None
    symptom: str | None = None
    urgency: str | None = None
    busy: list[str] = []
    skip: Annotated[int, Field(ge=0)] = 0  # leading plan steps already dispatched
    positions: dict[str, dict[str, float]] = {}  # current positions of patrolling units


def _waypoint(unit, km, index):
    """A deterministic point ``km`` away from the station for a patrol loop."""
    bearing = math.radians((zlib.crc32(unit["id"].encode()) % 360 + index * 137) % 360)
    return {"lat": unit["lat"] + km * math.cos(bearing) / 111.0,
            "lon": unit["lon"] + km * math.sin(bearing) / (111.0 * math.cos(math.radians(unit["lat"])))}


async def patrols(center, radius_km=5.0, limit=14):
    """Road loops station -> waypoint -> station for units near ``center`` (demo effect, not real patrols)."""
    near = sorted((u for u in fleet() if _km(u, center) <= radius_km), key=lambda u: _km(u, center))[:limit]
    async def loop(unit):
        point = _waypoint(unit, 1.2, len(unit["id"]))
        out, back = await asyncio.gather(route(unit, point), route(point, unit))
        return {"id": unit["id"], "path": out["path"] + back["path"][1:],
                "duration_s": out["duration_s"] + back["duration_s"], "source": out["source"]}
    return await asyncio.gather(*(loop(u) for u in near))


router = APIRouter(prefix="/api/map")


@router.get("/patrols")
async def get_patrols(lat: Annotated[float, Query(ge=37.0, le=38.0)], lon: Annotated[float, Query(ge=126.5, le=127.5)]):
    return {"patrols": await patrols({"lat": lat, "lon": lon}), "note": "시연용 순찰 경로 (실제 차량 위치 아님)"}


@router.get("/fleet")
def get_fleet():
    return {"units": fleet(), "attribution": stations()[1], "routing": provider()}


@router.post("/dispatch")
async def dispatch(body: DispatchRequest):
    try:
        steps = plan(body.dispatch, body.subtype, body.symptom, body.urgency)
    except ValueError:
        raise HTTPException(422, "unknown_dispatch") from None
    incident = {"lat": body.lat, "lon": body.lon}
    return {"incident": incident, "plan": steps,
            "assignments": await assign(incident, steps[body.skip:], body.busy,
                                        {k: {"lat": v["lat"], "lon": v["lon"]} for k, v in body.positions.items()
                                         if {"lat", "lon"} <= v.keys()})}
