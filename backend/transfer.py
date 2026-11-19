"""Transfer hospital: NMC emergency-care data + the KB transfer rule + a Zoro choice among the candidates.

Hospitals and live ER beds come from 국립중앙의료원 전국 응급의료기관 정보 조회 서비스 (data.go.kr,
NMC_SERVICE_KEY): getEgytListInfoInqire (level, location), getEmrrmRltmUsefulSckbdInfoInqire (hvec ER beds),
getSrsillDissAceptncPosblInfoInqire (MKioskTy serious-disease acceptance). Endpoint use and the MKioskTy
meanings follow ermct-backend (app/services/ermct_client.py, procedure_groups.py).

The rule quotes 「119구급대원 현장응급처치 표준지침(2023)」 이송대상기관 선정 및 결정 (p.54-55). Zoro picks one of
the rule-eligible, nearest candidates with the released decision head: this is not a trained task.
"""

import asyncio
import math
import os
import time
from urllib.parse import unquote
import xml.etree.ElementTree as ET

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .call_area import _dongs
from .routing import _setting, route


_BASE = "https://apis.data.go.kr/B552657/ErmctInfoInqireService/"
# Fallback when the key is not registered for the emergency service: hospital directory (level + location).
_DIRECTORY = "https://apis.data.go.kr/B552657/HsptlAsembySearchService/getHsptlMdcncListInfoInqire"
# Live capacity and equipment fields of getEmrrmRltmUsefulSckbdInfoInqire (ermct-backend ermct_client.py).
_BEDS = {"hvec": "응급실", "hvoc": "수술실", "hvicc": "일반중환자실", "hvcc": "신경중환자실", "hvccc": "흉부중환자실"}
_EQUIPMENT = {"hvctayn": "CT", "hvmriayn": "MRI", "hvangioayn": "혈관촬영기", "hvventiayn": "인공호흡기", "hvecmoayn": "ECMO"}
_TTL_S = 120.0
_CACHE: dict = {}
_KB = "119구급대원 현장응급처치 표준지침(2023) · 이송대상기관 선정 및 결정"
# NMC level names, strongest first.
LEVELS = ["권역응급의료센터", "전문응급의료센터", "지역응급의료센터", "지역응급의료기관"]
MKIOSK = {"MKioskTy1": "심근경색 재관류", "MKioskTy2": "뇌경색 재관류", "MKioskTy3": "뇌출혈 수술",
          "MKioskTy4": "대동맥 응급", "MKioskTy16": "분만", "MKioskTy17": "산과 수술", "MKioskTy19": "중증화상",
          "MKioskTy20": "사지접합"}


def transfer_rule(subtype=None, symptom=None, urgency=None):
    """KB rule for the committed EMS decisions: {category, min_level, prefer, text, source}."""
    if subtype == "심정지" or symptom == "심정지·호흡정지":
        return {"category": "심폐정지", "min_level": "지역응급의료기관", "prefer": [],
                "text": "가장 가까운 지역응급의료기관 이상 의료기관으로 이송", "source": f"{_KB} 3. 심폐정지 환자 (p.55)"}
    if symptom == "마비":
        return {"category": "뇌졸중 의심", "min_level": "지역응급의료센터", "prefer": ["MKioskTy2", "MKioskTy3"],
                "text": "가까운 지역응급의료센터 이상, 혈전용해(재관류) 가능 기관 우선",
                "source": f"{_KB} 4. 뇌졸중의증 환자 (p.55)"}
    if subtype in ("부상", "사고") and urgency == "상":
        return {"category": "중증외상", "min_level": "권역응급의료센터", "prefer": [],
                "text": "중증외상 진료가 가능한 가까운 권역외상센터 (외상센터 목록 미연동: 권역응급의료센터로 대체)",
                "source": f"{_KB} 1. 외상 환자 · 중증외상환자 (p.54)"}
    if subtype in ("부상", "사고"):
        return {"category": "경증 외상", "min_level": "지역응급의료기관", "prefer": [],
                "text": "가장 가까운 지역응급의료기관", "source": f"{_KB} 1. 외상 환자 · 경증 외상환자 (p.54)"}
    if symptom == "흉통·가슴불편감·심계항진" and urgency == "상":
        return {"category": "비외상 중증 (흉통)", "min_level": "지역응급의료센터", "prefer": ["MKioskTy1"],
                "text": "가까운 지역응급의료센터 이상, 심근경색 재관류 가능 기관 우선",
                "source": f"{_KB} 2. 비외상 환자 · 중증응급환자 (p.55)"}
    if subtype == "임산부":
        return {"category": "임산부", "min_level": "지역응급의료센터" if urgency == "상" else "지역응급의료기관",
                "prefer": ["MKioskTy16", "MKioskTy17"], "text": "분만·산과 수술 가능 기관 우선",
                "source": f"{_KB} 2. 비외상 환자 (p.55) · 수용 가능 질환은 NMC 중증질환 정보"}
    if urgency == "상":
        return {"category": "비외상 중증", "min_level": "지역응급의료센터", "prefer": [],
                "text": "가까운 지역응급의료센터 이상 의료기관", "source": f"{_KB} 2. 비외상 환자 · 중증응급환자 (p.55)"}
    return {"category": "경증 응급", "min_level": "지역응급의료기관", "prefer": [],
            "text": "가장 가까운 지역응급의료기관", "source": f"{_KB} 2. 비외상 환자 · 경증응급환자 (p.55)"}


def _key():
    raw = _setting("NMC_SERVICE_KEY")
    return unquote(raw) if raw else ""  # accepts either the encoding or the decoding form


async def _items(client, operation, params, base=_BASE):
    response = await client.get(base + operation if base == _BASE else base, params={**params, "serviceKey": _key()})
    response.raise_for_status()
    if b"<!DOCTYPE" in response.content[:200] or b"<!ENTITY" in response.content:
        raise ValueError("unexpected XML")
    root = ET.fromstring(response.content)
    code = root.findtext(".//resultCode")
    if code not in (None, "00"):
        raise ValueError(f"NMC result {code}")
    return [{child.tag: (child.text or "").strip() for child in item} for item in root.iter("item")]


def _int(value):
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


async def _directory(client, district):
    """Emergency-care hospitals from the hospital directory: level and location only, no live beds."""
    rows = []
    for kind in ("A", "B"):  # 종합병원, 병원
        rows += await _items(client, "", {"Q0": "서울특별시", "Q1": district, "QZ": kind, "numOfRows": "200", "pageNo": "1"},
                             base=_DIRECTORY)
    out = []
    for h in rows:
        level = h.get("dutyEmclsName") or ""
        if h.get("dutyEryn") != "1" or level not in LEVELS or not h.get("wgs84Lat"):
            continue
        out.append({"hpid": h["hpid"], "name": h.get("dutyName", h["hpid"]), "address": h.get("dutyAddr", ""),
                    "level": level, "district": district, "lat": float(h["wgs84Lat"]), "lon": float(h["wgs84Lon"]),
                    "er_beds": None, "beds": {}, "equipment": [], "beds_at": None, "accepts": [], "messages": [],
                    "data": "directory"})
    return out


async def _district(client, district):
    cached = _CACHE.get(district)
    if cached and time.monotonic() - cached[0] < _TTL_S:
        return cached[1]
    try:
        out = await _live_district(client, district)
    except (httpx.HTTPError, ValueError, ET.ParseError):
        out = await _directory(client, district)
    _CACHE[district] = (time.monotonic(), out)
    return out


async def _live_district(client, district):
    stage = {"STAGE1": "서울특별시", "STAGE2": district}
    listing, beds, serious = await asyncio.gather(
        _items(client, "getEgytListInfoInqire", {"Q0": "서울특별시", "Q1": district, "numOfRows": "100", "pageNo": "1"}),
        _items(client, "getEmrrmRltmUsefulSckbdInfoInqire", {**stage, "numOfRows": "100", "pageNo": "1"}),
        _items(client, "getSrsillDissAceptncPosblInfoInqire", {**stage, "SM_TYPE": "1", "numOfRows": "100", "pageNo": "1"}))
    beds_by = {b.get("hpid"): b for b in beds}
    serious_by = {s.get("hpid"): s for s in serious}
    out = []
    for h in listing:
        hpid = h.get("hpid")
        level = h.get("dutyEmclsName") or ""
        if not hpid or level not in LEVELS or not h.get("wgs84Lat"):
            continue
        bed = beds_by.get(hpid, {})
        accepts = [name for code, name in MKIOSK.items() if (serious_by.get(hpid, {}).get(code) or "").startswith("Y")]
        out.append({"hpid": hpid, "name": h.get("dutyName", hpid), "address": h.get("dutyAddr", ""), "level": level,
                    "district": district, "lat": float(h["wgs84Lat"]), "lon": float(h["wgs84Lon"]),
                    "er_beds": _int(bed.get("hvec")),
                    "beds": {name: _int(bed.get(code)) for code, name in _BEDS.items() if _int(bed.get(code)) is not None},
                    "equipment": [name for code, name in _EQUIPMENT.items() if (bed.get(code) or "").upper().startswith("Y")],
                    "beds_at": bed.get("hvidate"), "accepts": accepts, "messages": [], "data": "live"})
    return out


async def _messages(client, hospital):
    """ER / serious-disease messages (e.g. 수용 불가 안내) for one hospital; empty on any failure."""
    if hospital.get("data") != "live":
        return []
    try:
        rows = await _items(client, "getEmrrmSrsillDissMsgInqire", {"HPID": hospital["hpid"], "numOfRows": "10", "pageNo": "1"})
    except (httpx.HTTPError, ValueError, ET.ParseError):
        return []
    texts = [r.get("symBlkMsg", "").strip() for r in rows if r.get("symBlkMsg", "").strip()]
    return list(dict.fromkeys(texts))[:3]  # the same notice is often posted once per department


def _neighbors(point, count=3):
    """The incident's district plus the nearest other districts by dong-centre distance."""
    best = {}
    for (district, _), feature in _dongs().items():
        lon, lat = feature["properties"]["center"]
        d = math.hypot((lat - point["lat"]) * 111, (lon - point["lon"]) * 88)
        best[district] = min(best.get(district, 1e9), d)
    return [d for d, _ in sorted(best.items(), key=lambda kv: kv[1])[:count]]


async def candidates(point, rule, limit=4):
    if not _key():
        return "unconfigured", []
    try:
        async with httpx.AsyncClient(timeout=8.0, trust_env=False) as client:
            groups = await asyncio.gather(*(_district(client, d) for d in _neighbors(point)))
            floor = LEVELS.index(rule["min_level"])
            # Rule level first; a live count of 0 ER beds excludes, an unknown count (directory data) does not.
            eligible = [dict(h) for g in groups for h in g if LEVELS.index(h["level"]) <= floor and h["er_beds"] != 0]
            eligible.sort(key=lambda h: (-sum(1 for p in rule["prefer"] if MKIOSK[p] in h["accepts"]),
                                         math.hypot((h["lat"] - point["lat"]) * 111, (h["lon"] - point["lon"]) * 88)))
            picked = eligible[:limit]
            routes, messages = await asyncio.gather(asyncio.gather(*(route(point, h) for h in picked)),
                                                    asyncio.gather(*(_messages(client, h) for h in picked)))
    except (httpx.HTTPError, ValueError, ET.ParseError):
        return "unavailable", []
    for h, r, m in zip(picked, routes, messages):
        h["route"], h["messages"] = r, m
    data = "live" if picked and all(h["data"] == "live" for h in picked) else "directory"
    return data, sorted(picked, key=lambda h: h["route"]["duration_s"])


def _criterion(h, rule):
    """Option text for the Zoro choice: level, road ETA, live beds, procedures, equipment and messages."""
    parts = [h["level"], f"도착 약 {round(h['route']['duration_s'] / 60)}분 ({h['route']['distance_m'] / 1000:.1f}km)"]
    if h["beds"]:
        parts.append("가용 병상 " + ", ".join(f"{k} {v}" for k, v in h["beds"].items()))
    elif h["er_beds"] is None:
        parts.append("실시간 병상 정보 없음")
    if h["accepts"]:
        parts.append("수용 가능 수술·처치: " + ", ".join(h["accepts"]))
    if h["equipment"]:
        parts.append("가용 장비: " + ", ".join(h["equipment"]))
    if h["messages"]:
        parts.append("병원 메시지: " + " / ".join(h["messages"]))
    return f"{h['name']} · " + " · ".join(parts)


async def zoro_choice(turns, rule, hospitals):
    """Released-head choice over the candidates via the decision server's /v1/choose."""
    url = os.getenv("SENTINEL_ZORO_URL", "").strip().rstrip("/")
    if not url or not hospitals:
        return None
    question = {"instructions": (f"구급 환자를 이송할 병원을 고르세요. 환자 분류: {rule['category']}. "
                                 f"이송 원칙: {rule['text']}. 도착 시간이 짧고 응급실 병상이 있는 병원을 우선합니다."),
                "criteria": {h["hpid"]: _criterion(h, rule) for h in hospitals}}
    try:
        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
            response = await client.post(url + "/v1/choose", json={"turns": turns, "question": question})
            response.raise_for_status()
            data = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    probabilities = data["results"]["probabilities"]
    return {"probabilities": probabilities, "choice": max(probabilities, key=probabilities.get),
            "latency_ms": data.get("latency_ms"), "question": question}


class TransferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lat: float = Field(ge=37.0, le=38.0)
    lon: float = Field(ge=126.5, le=127.5)
    turns: list[dict] = Field(min_length=1, max_length=120)
    subtype: str | None = None
    symptom: str | None = None
    urgency: str | None = None


router = APIRouter(prefix="/api/map")


@router.post("/transfer")
async def transfer(body: TransferRequest):
    turns = [{"speaker": 1 if t.get("speaker") else 0, "text": str(t.get("text", ""))[:2000]}
             for t in body.turns if str(t.get("text", "")).strip()]
    if not turns:
        raise HTTPException(422, "empty_turns")
    point = {"lat": body.lat, "lon": body.lon}
    rule = transfer_rule(body.subtype, body.symptom, body.urgency)
    data, hospitals = await candidates(point, rule)
    status = data if data in ("unconfigured", "unavailable") else "ok"
    choice = await zoro_choice(turns, rule, hospitals) if status == "ok" else None
    for h in hospitals:
        h["context"] = _criterion(h, rule)
    source = ("국립중앙의료원 전국 응급의료기관 정보 조회 서비스 (실시간 병상·수술·장비·메시지)" if data == "live"
              else "국립중앙의료원 병·의원 찾기 서비스 (응급의료기관 등급·위치, 실시간 병상 미연동)")
    return {"status": status, "data": data, "rule": rule, "hospitals": hospitals, "zoro": choice, "source": source}
