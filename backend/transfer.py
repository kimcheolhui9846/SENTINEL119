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
# NMC serious-disease acceptance (MKioskTy) grouped as in ermct-backend app/procedure_groups.py; a group is
# available when any of its codes is "Y". Codes 5, 6 and 28 have no group there and are left out.
PROCEDURES = {
    "심근경색 재관류(PCI)": ["MKioskTy1"], "뇌경색 재관류": ["MKioskTy2"], "뇌출혈 수술": ["MKioskTy3"],
    "대동맥 응급": ["MKioskTy4"], "복부 응급수술": ["MKioskTy7", "MKioskTy8", "MKioskTy9"],
    "장중첩·장폐색": ["MKioskTy10"], "위장관 내시경": ["MKioskTy11", "MKioskTy12"],
    "기관지 내시경": ["MKioskTy13", "MKioskTy14"], "신생아 집중치료": ["MKioskTy15"],
    "분만·산과 수술": ["MKioskTy16", "MKioskTy17", "MKioskTy18"], "중증화상": ["MKioskTy19"],
    "사지접합": ["MKioskTy20", "MKioskTy21"], "응급투석": ["MKioskTy22", "MKioskTy23"],
    "정신과 폐쇄병동": ["MKioskTy24"], "안과 응급": ["MKioskTy25"], "영상의학 혈관중재": ["MKioskTy26", "MKioskTy27"],
}
_RULE_TEXT = {
    "심폐정지": ("지역응급의료기관", "가장 가까운 지역응급의료기관 이상 의료기관으로 이송", "3. 심폐정지 환자 (p.55)"),
    "뇌졸중 의심": ("지역응급의료센터", "가까운 지역응급의료센터 이상, 혈전용해(재관류) 가능 기관 우선", "4. 뇌졸중의증 환자 (p.55)"),
    "중증외상": ("권역응급의료센터", "중증외상 진료가 가능한 가까운 권역외상센터 (외상센터 목록 미연동: 권역응급의료센터로 대체)",
                "1. 외상 환자 · 중증외상환자 (p.54)"),
    "경증 외상": ("지역응급의료기관", "가장 가까운 지역응급의료기관", "1. 외상 환자 · 경증 외상환자 (p.54)"),
    "비외상 중증": ("지역응급의료센터", "가까운 지역응급의료센터 이상 의료기관", "2. 비외상 환자 · 중증응급환자 (p.55)"),
    "경증 응급": ("지역응급의료기관", "가장 가까운 지역응급의료기관", "2. 비외상 환자 · 경증응급환자 (p.55)"),
}
# Main symptom -> (KB category, procedures to prefer). The symptom decides; urgency only splits trauma.
_BY_SYMPTOM = {
    "심정지·호흡정지": ("심폐정지", []),
    "마비": ("뇌졸중 의심", ["뇌경색 재관류", "뇌출혈 수술"]),
    "의식저하·실신": ("비외상 중증", ["뇌경색 재관류", "뇌출혈 수술", "심근경색 재관류(PCI)"]),
    "경련·발작": ("비외상 중증", ["뇌출혈 수술"]),
    "흉통·가슴불편감·심계항진": ("비외상 중증", ["심근경색 재관류(PCI)", "대동맥 응급"]),
    "호흡곤란·기도이물": ("비외상 중증", ["기관지 내시경"]),
    "출혈(토혈·혈변·객혈·코피 등)": ("비외상 중증", ["위장관 내시경", "영상의학 혈관중재"]),
    "분만진통·질출혈": ("비외상 중증", ["분만·산과 수술", "신생아 집중치료"]),
    "복통·구토·설사": ("경증 응급", ["복부 응급수술", "장중첩·장폐색", "위장관 내시경"]),
    "외상(열상·골절·화상 등)": ("경증 외상", ["사지접합", "중증화상"]),
}


def transfer_rule(subtype=None, symptom=None, urgency=None):
    """KB transfer rule chosen by the main symptom (falls back to the subtype when no symptom is known)."""
    if subtype == "심정지":
        category, prefer = "심폐정지", []
    elif symptom in _BY_SYMPTOM:
        category, prefer = _BY_SYMPTOM[symptom]
    elif subtype == "임산부":
        category, prefer = "비외상 중증", ["분만·산과 수술", "신생아 집중치료"]
    elif subtype in ("부상", "사고"):
        category, prefer = "경증 외상", ["사지접합", "중증화상"]
    else:
        category, prefer = "경증 응급", []
    if category == "경증 외상" and urgency == "상":
        category = "중증외상"
    min_level, text, section = _RULE_TEXT[category]
    return {"category": category, "symptom": symptom, "min_level": min_level, "prefer": prefer, "text": text,
            "source": f"{_KB} {section}" + (" · 우선 처치는 NMC 중증질환 수용 정보" if prefer else "")}


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
        row = serious_by.get(hpid, {})
        accepts = [name for name, codes in PROCEDURES.items() if any((row.get(c) or "").startswith("Y") for c in codes)]
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
            floor = LEVELS.index(rule["min_level"])
            # Rule level first; a live count of 0 ER beds excludes, an unknown count (directory data) does not.
            # Widen from the 3 nearest districts to 8 when no hospital nearby meets the level (e.g. 권역 centres).
            for count in (3, 8):
                groups = await asyncio.gather(*(_district(client, d) for d in _neighbors(point, count)))
                eligible = [dict(h) for g in groups for h in g if LEVELS.index(h["level"]) <= floor and h["er_beds"] != 0]
                if eligible:
                    break
            eligible.sort(key=lambda h: (-sum(1 for p in rule["prefer"] if p in h["accepts"]),
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


# Information groups in the option text; the explanation removes one group at a time.
GROUPS = {"eta": "도착 시간", "beds": "가용 병상", "procedures": "필요 처치 가능 여부", "equipment": "가용 장비",
          "messages": "병원 메시지"}


def _criterion(h, rule, drop=()):
    """Option text for the Zoro choice: level, road ETA, live beds, procedures, equipment and messages."""
    parts = [h["level"]]
    if "eta" not in drop:
        parts.append(f"도착 약 {round(h['route']['duration_s'] / 60)}분 ({h['route']['distance_m'] / 1000:.1f}km)")
    if "beds" in drop:
        pass
    elif h["beds"]:
        parts.append("가용 병상 " + ", ".join(f"{k} {v}" for k, v in h["beds"].items()))
    elif h["er_beds"] is None:
        parts.append("실시간 병상 정보 없음")
    needed = [p for p in rule["prefer"] if p in h["accepts"]] if "procedures" not in drop else []
    missing = [p for p in rule["prefer"] if p not in h["accepts"]] if "procedures" not in drop else []
    if needed:
        parts.append("이 환자에게 필요한 처치 가능: " + ", ".join(needed))
    if missing and h["data"] == "live":
        parts.append("필요 처치 불가·미제공: " + ", ".join(missing))
    others = [a for a in h["accepts"] if a not in rule["prefer"]] if "procedures" not in drop else []
    if others:
        parts.append("그 밖의 수용 가능 수술·처치: " + ", ".join(others))
    if h["equipment"] and "equipment" not in drop:
        parts.append("가용 장비: " + ", ".join(h["equipment"]))
    if h["messages"] and "messages" not in drop:
        parts.append("병원 메시지: " + " / ".join(h["messages"]))
    return f"{h['name']} · " + " · ".join(parts)


async def zoro_choice(turns, rule, hospitals, drop=()):
    """Released-head choice over the candidates via the decision server's /v1/choose."""
    url = os.getenv("SENTINEL_ZORO_URL", "").strip().rstrip("/")
    if not url or not hospitals:
        return None
    need = f" 필요한 처치: {', '.join(rule['prefer'])}." if rule["prefer"] else ""
    question = {"instructions": (f"구급 환자를 이송할 병원을 고르세요. 주증상: {rule.get('symptom') or '미상'}. "
                                 f"환자 분류: {rule['category']}. 이송 원칙: {rule['text']}.{need} "
                                 "필요한 처치가 가능하고, 도착 시간이 짧고, 응급실 병상이 있고, 수용 제한 메시지가 없는 병원을 우선합니다."),
                "criteria": {h["hpid"]: _criterion(h, rule, drop) for h in hospitals}}
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


def _facts(chosen, other, rule):
    """Plain comparison of the chosen hospital with the runner-up, from the same data Zoro read."""
    out = []
    dt = round((chosen["route"]["duration_s"] - other["route"]["duration_s"]) / 60)
    out.append(f"도착 {abs(dt)}분 {'더 빠름' if dt < 0 else '더 걸림'}" if dt else "도착 시간 비슷함")
    a, b = chosen.get("er_beds"), other.get("er_beds")
    if a is not None and b is not None and a != b:
        out.append(f"응급실 병상 {abs(a - b)}개 {'더 많음' if a > b else '더 적음'} ({a} vs {b})")
    need_a = [p for p in rule["prefer"] if p in chosen["accepts"]]
    need_b = [p for p in rule["prefer"] if p in other["accepts"]]
    extra = [p for p in need_a if p not in need_b]
    if extra:
        out.append("필요 처치 추가 가능: " + ", ".join(extra))
    lacking = [p for p in need_b if p not in need_a]
    if lacking:
        out.append("2순위만 가능한 필요 처치: " + ", ".join(lacking))
    if not chosen["messages"] and other["messages"]:
        out.append(f"제한 공지 없음 (2순위: {other['messages'][0][:40]})")
    elif chosen["messages"] and not other["messages"]:
        out.append(f"제한 공지 있음: {chosen['messages'][0][:40]}")
    if LEVELS.index(chosen["level"]) < LEVELS.index(other["level"]):
        out.append(f"더 높은 등급 ({chosen['level']})")
    return out


def _summary(best, runner, effects, p, rule):
    """Two or three Korean sentences from the measured influence and the runner-up comparison (no generation)."""
    out = [f"이송 병원으로 {best['name']}을(를) 선택했습니다 ({round(p * 100)}%)."]
    top = effects[0] if effects and effects[0]["delta"] > 0.02 else None
    if top:
        detail = ""
        if top["group"] == "병원 메시지":
            if runner and runner["messages"] and not best["messages"]:
                detail = f" 2순위 {runner['name']}에 '{runner['messages'][0][:30]}' 공지가 있었습니다."
            elif best["messages"]:
                detail = f" 선택한 병원의 공지: '{best['messages'][0][:30]}'."
        elif top["group"] == "가용 병상" and best.get("er_beds") is not None:
            detail = f" 선택한 병원의 응급실 가용 병상은 {best['er_beds']}개입니다."
        elif top["group"] == "필요 처치 가능 여부":
            ok = [x for x in rule["prefer"] if x in best["accepts"]]
            detail = f" 선택한 병원은 {', '.join(ok)}이(가) 가능합니다." if ok else ""
        elif top["group"] == "가용 장비" and best["equipment"]:
            missing = [e for e in (runner or {}).get("equipment", []) if e not in best["equipment"]]
            fewer = [e for e in best["equipment"] if runner and e not in runner["equipment"]]
            detail = f" 선택한 병원은 {', '.join(best['equipment'])}을(를) 쓸 수 있습니다." + (
                f" 2순위에는 {', '.join(fewer)}이(가) 없습니다." if fewer else "")
        elif top["group"] == "도착 시간":
            detail = f" 도착까지 약 {round(best['route']['duration_s'] / 60)}분입니다."
        out.append(f"가장 크게 작용한 정보는 {top['group']}입니다. 이 정보를 빼면 선택 확률이 "
                   f"{round(top['delta'] * 100)}%p 내려갑니다.{detail}")
    else:
        out.append("어느 한 정보가 결정적이지 않았고, 여러 정보를 함께 반영한 선택입니다.")
    if runner:
        facts = _facts(best, runner, rule)
        if facts:
            out.append(f"2순위 {runner['name']}와(과) 비교하면 " + ", ".join(facts) + ".")
    return out


async def explain(turns, rule, hospitals, full):
    """Leave-one-group-out: how much the chosen hospital's probability drops without each information group."""
    chosen = full["choice"]
    ablated = await asyncio.gather(*(zoro_choice(turns, rule, hospitals, drop=(g,)) for g in GROUPS))
    effects = []
    for group, result in zip(GROUPS, ablated):
        if result is None:
            continue
        effects.append({"group": GROUPS[group], "without": result["probabilities"][chosen],
                        "delta": full["probabilities"][chosen] - result["probabilities"][chosen],
                        "choice_without": result["choice"]})
    effects.sort(key=lambda e: -e["delta"])
    ranked = sorted(hospitals, key=lambda h: -full["probabilities"][h["hpid"]])
    best = next(h for h in hospitals if h["hpid"] == chosen)
    runner = ranked[1] if len(ranked) > 1 else None
    return {"chosen": best["name"], "p": full["probabilities"][chosen], "effects": effects,
            "summary": _summary(best, runner, effects, full["probabilities"][chosen], rule),
            "runner_up": runner["name"] if runner else None,
            "runner_up_p": full["probabilities"][runner["hpid"]] if runner else None,
            "facts": _facts(best, runner, rule) if runner else [],
            "method": "정보 묶음을 하나씩 뺀 선택지로 Zoro를 다시 호출해, 선택한 병원의 확률이 줄어든 만큼을 영향으로 표시"}


async def zoro_reason(turns, rule, hospitals, question):
    """Zoro-1 reasoned action on the same hospital question (decision server /v1/reason); None on failure."""
    url = os.getenv("SENTINEL_ZORO_URL", "").strip().rstrip("/")
    if not url:
        return None
    try:
        async with httpx.AsyncClient(timeout=180.0, trust_env=False) as client:
            response = await client.post(url + "/v1/reason", json={"turns": turns, "question": question,
                                                                   "max_new_tokens": 384})
            response.raise_for_status()
            data = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    names = {h["hpid"]: h["name"] for h in hospitals}
    choice = max(data["reasoned"], key=data["reasoned"].get)
    return {"text": data["reasoning"], "finished": data["reasoning_finished"], "tokens": data["reasoning_tokens"],
            "ms": data["reasoning_ms"], "direct": data["direct"], "reasoned": data["reasoned"], "choice": choice,
            "choice_name": names.get(choice, choice)}


class TransferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lat: float = Field(ge=37.0, le=38.0)
    lon: float = Field(ge=126.5, le=127.5)
    turns: list[dict] = Field(min_length=1, max_length=120)
    subtype: str | None = None
    symptom: str | None = None
    urgency: str | None = None
    explain: bool = False
    reason: bool = False  # also run Zoro-1 reasoned action (slow, experimental)


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
    reason = None
    if body.explain and choice:
        if body.reason:
            reason, reasoned = await asyncio.gather(explain(turns, rule, hospitals, choice),
                                                    zoro_reason(turns, rule, hospitals, choice["question"]))
        else:
            reason, reasoned = await explain(turns, rule, hospitals, choice), None
        reason["reasoned"] = reasoned
    for h in hospitals:
        h["context"] = _criterion(h, rule)
    source = ("국립중앙의료원 전국 응급의료기관 정보 조회 서비스 (실시간 병상·수술·장비·메시지)" if data == "live"
              else "국립중앙의료원 병·의원 찾기 서비스 (응급의료기관 등급·위치, 실시간 병상 미연동)")
    return {"status": status, "data": data, "rule": rule, "hospitals": hospitals, "zoro": choice, "source": source,
            "reason": reason}
