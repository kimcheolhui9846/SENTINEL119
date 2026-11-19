"""Synthetic Seoul call replays. Only the current frame leaves this provider.

SENTINEL_SCENARIO_DIRS (os.pathsep-separated) adds turn-wise call files ({id, title, summary,
turns: [{speaker: 0 dispatcher | 1 caller, text}], gold?, address?}). They run in live mode only: there is no
reviewed fixture, so the model's confidence decides WAIT/COMMIT. Files under a "private" directory are
real AI Hub calls for local demos; keep them out of the repository.
"""

import json
import os
from pathlib import Path

from .call_area import find_area, virtual_resources


_CATALOG = {
    "staged": ("성수역 광장 신고", "발화가 쌓이며 출동유형, 차량, 병원 후보가 차례로 결정됩니다.", [
        ("신고자", "여기 사람이 모여 있어요. 도와주세요."),
        ("신고자", "성수역 2번 출구 광장이에요. 한 분이 넘어져 구급차가 필요해요."),
        ("신고자", "의식이 있고 제 말에 대답해요. 숨은 평소처럼 쉬는데 넘어진 뒤 다리가 계속 아프다고 해요."),
        ("신고자", "위치를 정정할게요. 2번이 아니라 3번 출구 앞입니다."),
    ]),
    "reroute": ("강변 진입로 통제", "도로 상황이 바뀌면서 두 가상 차량의 도착 예정시간이 역전됩니다.", [
        ("신고자", "뚝섬유원지 북문 앞입니다. 다친 분이 있어서 구급차를 요청해요."),
        ("수보요원", "현재 위치를 확인했습니다. 차량별 접근 경로가 갱신됐습니다."),
        ("신고자", "북문에서 기다리고 있어요. 구급차가 들어올 곳을 비워 두었습니다."),
    ]),
    "outage": ("병원 수용 정보 지연", "병원 정보 조회가 중단돼도 차량 판단은 이어집니다.", [
        ("신고자", "서울숲 방문자센터 앞입니다. 구급차가 필요해요."),
        ("수보요원", "방문자센터 앞을 확인했습니다. 병원 수용 정보를 조회 중입니다."),
        ("신고자", "같은 위치에서 기다리고 있습니다."),
    ]),
}


def _call_files():
    calls = {}
    for root in filter(None, os.getenv("SENTINEL_SCENARIO_DIRS", "").split(os.pathsep)):
        for path in sorted(Path(root).glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            turns = [("수보요원" if turn["speaker"] == 0 else "신고자", turn["text"]) for turn in data["turns"]]
            calls[data["id"]] = {"title": data["title"], "summary": data.get("summary", ""), "turns": turns,
                                 "gold": data.get("gold"), "private": "private" in path.parts,
                                 "area": find_area(data.get("address"))}
    return calls


def list_scenarios():
    synthetic = [{"id": key, "title": title, "summary": summary, "turn_count": len(turns), "live_only": False}
                 for key, (title, summary, turns) in _CATALOG.items()]
    calls = [{"id": key, "title": call["title"], "summary": call["summary"], "turn_count": len(call["turns"]),
              "live_only": True, "private": call["private"],
              "area": f"{call['area']['district']} {call['area']['dong']}" if call["area"] else None}
             for key, call in _call_files().items()]
    return synthetic + calls


_OBSERVED_AT = "2026-11-20T09:00:00+09:00"
_HOSPITAL_STATUS = ("available", "available", "unavailable")


def _call_frame(call, step):
    """Turn replay around the call's 법정동 with virtual units; nothing here reviews the labels.

    The carrier position is known from the first turn, so the area is shown from step 1."""
    speaker, text = call["turns"][step - 1] if step else ("", "")
    area = call["area"]
    if area is None:  # no Seoul address: keep the staged virtual candidates without a map position
        staged = get_frame("staged", 3)
        return {**staged, "speaker": speaker, "text": text, "supports": {}, "simulation_choices": {},
                "location": ("가상 위치 · 실제 주소 비공개", 1) if step else None,
                "external_status": "ok" if step else "unknown", "area": None, "units": []}
    units, hospitals = virtual_resources(area)
    source = "가상 차량 · 법정동 중심 기준 거리"
    vehicles = [{**_candidate(u["id"], u["name"], u["eta_min"], u["detail"], "available", source, _OBSERVED_AT),
                 "lat": u["lat"], "lon": u["lon"], "kind": u["kind"]} for u in units if u["kind"] == "ambulance"]
    hospital_rows = [{**_candidate(h["id"], h["name"], h["eta_min"], h["detail"], status if step else "unknown",
                                   "가상 병원 · 시연용 수용 정보", _OBSERVED_AT if step else None),
                      "lat": h["lat"], "lon": h["lon"]} for h, status in zip(hospitals, _HOSPITAL_STATUS)]
    return {"speaker": speaker, "text": text, "supports": {}, "simulation_choices": {},
            "location": (f"{area['district']} {area['dong']} 일대 (법정동)", 1) if step else None,
            "vehicles": vehicles, "hospitals": hospital_rows, "external_status": "ok" if step else "unknown",
            "area": area, "units": units}


def _candidate(key, name, minutes, detail, status, source, source_at):
    return {"id": key, "name": name, "eta_min": minutes, "detail": detail,
            "availability": status, "source": source, "source_at": source_at}


def get_frame(scenario_id, step, hospital_retry=False):
    """Return fresh, caller-owned containers. Step zero is the empty incident."""
    if scenario_id not in _CATALOG:
        call = _call_files()[scenario_id]
        if not 0 <= step <= len(call["turns"]):
            raise ValueError("stage_out_of_range")
        return _call_frame(call, step)
    _, _, turns = _CATALOG[scenario_id]
    if not 0 <= step <= len(turns):
        raise ValueError("stage_out_of_range")
    first = 2 if scenario_id == "staged" else 1
    dispatch_ready = step >= first
    # Only the caller's clinical assessment in the staged replay supports urgency.
    urgency_ready = scenario_id == "staged" and step >= 3
    support = {"dispatch": {"구급": [first]} if dispatch_ready else {},
               "urgency": {"중": [3]} if urgency_ready else {}}
    locations = {"staged": "성수역 2번 출구", "reroute": "뚝섬유원지 북문", "outage": "서울숲 방문자센터"}
    location = (locations[scenario_id], first) if dispatch_ready else None
    if scenario_id == "staged" and step == 4:
        location = ("성수역 3번 출구", 4)
    swapped = (scenario_id == "reroute" and step >= 2) or (scenario_id == "staged" and step == 4)
    vehicle_source = "가상 차량 · 시나리오 ETA"
    eta_observed_step = (2 if scenario_id == "reroute" else 4) if swapped else 0
    vehicle_source_at = f"2026-11-19T14:{eta_observed_step:02}:00+09:00"
    vehicles = [
        _candidate("A", "구급차 A", 12 if swapped else 3, "성동 거점 · 기본 구급 장비", "available", vehicle_source, vehicle_source_at),
        _candidate("B", "구급차 B", 4 if swapped else 9, "광진 거점 · 기본 구급 장비", "available", vehicle_source, vehicle_source_at),
    ]
    if not dispatch_ready:
        external = "unknown"
    elif scenario_id == "outage" and step == 2 and not hospital_retry:
        external = "timeout"
    elif scenario_id == "staged" and step < 3:
        external = "unknown"
    else:
        external = "ok"
    source = "가상 병원 · 시나리오 자료" if external != "timeout" else "가상 병원 · 시나리오 조회 시간 초과"
    observed_at = None
    if external == "ok":
        seconds = 30 if hospital_retry else 0
        observed_at = f"2026-11-19T14:{step:02}:{seconds:02}+09:00"
    hospitals = [
        _candidate("H1", "동부 시뮬레이션 병원", 8, "수용 현황 조회 대상", "unknown", source, None),
        _candidate("H2", "성동 시뮬레이션 병원", 11, "수용 정보 제공 대상", "available" if observed_at else "unknown", source, observed_at),
        _candidate("H3", "강변 시뮬레이션 병원", 16, "수용 현황 조회 대상", "unavailable" if observed_at else "unknown", source, observed_at),
    ]
    speaker, text = turns[step-1] if step else ("", "")
    return {"speaker": speaker, "text": text, "location": location, "supports": support,
            "vehicles": vehicles, "hospitals": hospitals, "external_status": external,
            "simulation_choices": {"dispatch": "구급", "urgency": "중",
                                   "vehicle": "B" if swapped else "A", "hospital": "H2"}}
