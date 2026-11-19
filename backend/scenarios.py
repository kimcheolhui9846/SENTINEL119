"""Synthetic Seoul call replays. Only the current frame leaves this provider."""


_CATALOG = {
    "staged": ("성수역 광장 신고", "발화가 쌓이며 출동유형, 차량, 병원 후보가 차례로 결정됩니다.", [
        ("신고자", "여기 사람이 모여 있어요. 도와주세요."),
        ("신고자", "성수역 2번 출구 광장이에요. 한 분이 넘어져 구급차가 필요해요."),
        ("신고자", "옆에서 계속 말을 걸고 있어요. 지금 구급대가 오고 있나요?"),
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


def list_scenarios():
    return [{"id": key, "title": title, "summary": summary, "turn_count": len(turns)}
            for key, (title, summary, turns) in _CATALOG.items()]


def _candidate(key, name, minutes, detail, status, source, minute=0):
    return {"id": key, "name": name, "eta_min": minutes, "detail": detail,
            "availability": status, "source": source,
            "source_at": f"2026-11-19T14:{minute:02}:00+09:00"}


def get_frame(scenario_id, step, hospital_retry=False):
    """Return fresh, caller-owned containers. Step zero is the empty incident."""
    _, _, turns = _CATALOG[scenario_id]
    if not 0 <= step <= len(turns):
        raise ValueError("stage_out_of_range")
    first = 2 if scenario_id == "staged" else 1
    dispatch_ready = step >= first
    urgency_ready = step >= (3 if scenario_id == "staged" else 2)
    support = {"dispatch": {"구급": [first]} if dispatch_ready else {},
               "urgency": {"중": [3 if scenario_id == "staged" else 2]} if urgency_ready else {}}
    locations = {"staged": "성수역 2번 출구", "reroute": "뚝섬유원지 북문", "outage": "서울숲 방문자센터"}
    location = (locations[scenario_id], first) if dispatch_ready else None
    if scenario_id == "staged" and step == 4:
        location = ("성수역 3번 출구", 4)
    swapped = (scenario_id == "reroute" and step >= 2) or (scenario_id == "staged" and step == 4)
    vehicle_source = "가상 차량 · 시나리오 ETA"
    vehicles = [
        _candidate("A", "구급차 A", 12 if swapped else 3, "성동 거점 · 기본 구급 장비", "available", vehicle_source, int(swapped)),
        _candidate("B", "구급차 B", 4 if swapped else 9, "광진 거점 · 기본 구급 장비", "available", vehicle_source, int(swapped)),
    ]
    if not dispatch_ready:
        external = "unknown"
    elif scenario_id == "outage" and step == 2 and not hospital_retry:
        external = "timeout"
    elif scenario_id == "staged" and step < 3:
        external = "unknown"
    else:
        external = "ok"
    source = "가상 병원 · 시나리오 자료" if external != "timeout" else "가상 병원 · 직전 저장 자료"
    hospitals = [
        _candidate("H1", "동부 시뮬레이션 병원", 8, "수용 현황 조회 대상", "unknown", source),
        _candidate("H2", "성동 시뮬레이션 병원", 11, "수용 정보 제공 대상", "available" if external == "ok" else "unknown", source, int(bool(hospital_retry))),
        _candidate("H3", "강변 시뮬레이션 병원", 16, "수용 현황 조회 대상", "unavailable" if external == "ok" else "unknown", source),
    ]
    speaker, text = turns[step-1] if step else ("", "")
    return {"speaker": speaker, "text": text, "location": location, "supports": support,
            "vehicles": vehicles, "hospitals": hospitals, "external_status": external,
            "simulation_choices": {"dispatch": "구급", "urgency": "중",
                                   "vehicle": "B" if swapped else "A", "hospital": "H2"}}
