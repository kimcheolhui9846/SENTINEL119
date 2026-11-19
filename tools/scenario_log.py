"""Run a scenario through the dispatch agent and write a submission log (JSONL, one event per line).

The agent is the demo backend itself, in process: each turn prefix goes to /api/compare/decide (the
SENTINEL_ZORO decision server, SFT and base heads), the turn-wise commit rule below opens WAIT/COMMIT per task,
committed decisions go to /api/map/dispatch (KB plan + road routes), and EMS calls go to /api/map/transfer
(KB transfer rule + NMC hospital data + Zoro hospital choice). This mirrors frontend/src/DispatchView.tsx.

Faults listed in the scenario file are injected for real: the decision server URL points at a closed port for
one turn, and the NMC district lookup raises a timeout for the first transfer requests. Both are marked
"injected" in the log.

    SENTINEL_ZORO_URL=http://127.0.0.1:8119 python tools/scenario_log.py docs/submission/scenarios/scenario-01.json \
        docs/submission/logs/scenario-01.jsonl
"""

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import transfer as transfer_module  # noqa: E402
from backend.call_area import find_area  # noqa: E402
from backend.drill_console import create_console  # noqa: E402

KST = timezone(timedelta(hours=9))
# Same values as frontend/src/CompareView.tsx (H4 thresholds tuned on dev, reward +1 / -2 / 0).
DISPATCH_THRESHOLD = {"ems": 0.66, "rescue": 0.86, "fire": 0.79, "other": 0.82}
SUBTYPE_THRESHOLD = {"ems_subtype": 0.86, "rescue_subtype": 0.68}
URGENCY_THRESHOLD = 0.82
SYMPTOM_THRESHOLD = 0.69
SUBTYPE_OF = {"ems": "ems_subtype", "rescue": "rescue_subtype"}
POLICY = "turn-wise threshold commit (H4, dev 83 calls, reward +1/-2/0); see frontend/src/CompareView.tsx"


def now():
    return datetime.now(KST).isoformat(timespec="milliseconds")


def top(dist):
    key = max(dist, key=dist.get)
    return key, dist[key]


def rounded(dist):
    return {k: round(v, 4) for k, v in dist.items()}


class Log:
    def __init__(self, path, run_id, scenario_id):
        self.path, self.base = path, {"run_id": run_id, "scenario_id": scenario_id}
        self.rows, self.seq = [], 0

    def __call__(self, event, **fields):
        self.seq += 1
        self.rows.append({**self.base, "seq": self.seq, "at": now(), "event": event, **fields})

    def write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.rows), encoding="utf-8")


def advance(commits, probs, turn, labels):
    """Commit each task once, the first time its top probability reaches the threshold (CompareView.advance)."""
    fresh = []

    def attempt(task, qid, threshold):
        if task in commits or qid not in probs:
            return
        key, p = top(probs[qid])
        needed = DISPATCH_THRESHOLD.get(key, threshold) if task == "dispatch" else threshold
        if p >= needed:
            commits[task] = {"question": qid, "key": key, "label": labels[qid].get(key, key), "p": round(p, 4),
                             "threshold": needed, "turn": turn}
            fresh.append(task)

    attempt("dispatch", "dispatch", 1.1)
    if "dispatch" in commits:  # the hierarchy opens only after the dispatch type is fixed
        sub = SUBTYPE_OF.get(commits["dispatch"]["key"])
        if sub:
            attempt("subtype", sub, SUBTYPE_THRESHOLD[sub])
        attempt("urgency", "urgency", URGENCY_THRESHOLD)
        if sub == "ems_subtype":
            attempt("symptom", "ems_symptom", SYMPTOM_THRESHOLD)
    return fresh


def tasks_view(commits, probs, labels):
    """Per task: COMMIT with the committed label, or WAIT with the current leader and the rule that applies."""
    view = {}
    for task, qid in (("dispatch", "dispatch"), ("subtype", SUBTYPE_OF.get(commits.get("dispatch", {}).get("key"))),
                      ("urgency", "urgency"),
                      ("symptom", "ems_symptom" if commits.get("dispatch", {}).get("key") == "ems" else None)):
        if task in commits:
            c = commits[task]
            view[task] = {"state": "COMMIT", "label": c["label"], "p": c["p"], "turn": c["turn"]}
        elif task != "dispatch" and "dispatch" not in commits:
            view[task] = {"state": "LOCKED", "reason": "출동유형 확정 전"}
        elif qid is None:
            view[task] = {"state": "N/A"}
        elif qid in probs:
            key, p = top(probs[qid])
            view[task] = {"state": "WAIT", "leader": labels[qid].get(key, key), "p": round(p, 4)}
    return view


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


async def run(scenario_path, out_path):
    scenario = json.loads(Path(scenario_path).read_text(encoding="utf-8"))
    sid = scenario["id"]
    log = Log(Path(out_path), f"{sid}-{datetime.now(KST):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}", sid)
    zoro_url = os.environ.get("SENTINEL_ZORO_URL", "")
    faults = scenario.get("faults", [])
    decision_faults = {f["turn"]: f for f in faults if f["target"] == "decision_server"}
    nmc_fault = next((f for f in faults if f["target"] == "nmc_api"), None)
    nmc_left = nmc_fault["first_requests"] if nmc_fault else 0

    real_district = transfer_module._district

    async def district(client, name):
        if nmc_left > 0:
            raise httpx.ConnectTimeout("injected NMC timeout")
        return await real_district(client, name)

    transfer_module._district = district
    area = find_area(scenario["address"], seed=sid)
    if area is None:
        raise SystemExit("scenario address is not a Seoul 법정동")
    point = area["center"]
    started = time.perf_counter()
    log("run_start", mode={"model": "live_model", "turns": "synthetic_script", "vehicles": "synthetic_world",
                           "routes": "live_api", "hospitals": "live_api"},
        git_commit=git_sha(), scenario={k: scenario[k] for k in ("title", "summary", "address", "expected")},
        config={"policy": POLICY, "dispatch_threshold": DISPATCH_THRESHOLD, "subtype_threshold": SUBTYPE_THRESHOLD,
                "urgency_threshold": URGENCY_THRESHOLD, "symptom_threshold": SYMPTOM_THRESHOLD,
                "decision_server": "SENTINEL_ZORO_URL (local tunnel)" if zoro_url else None},
        incident={"district": area["district"], "dong": area["dong"], "point": point,
                  "note": "주소의 법정동 안에서 시드 고정 무작위 지점 (발신 위치 대용)"},
        faults=faults)

    commits, probs, labels = {}, {}, {}
    orders, dispatch_key, transfer_key, model_logged = [], None, None, False
    transport = httpx.ASGITransport(app=create_console())
    async with httpx.AsyncClient(transport=transport, base_url="http://agent", timeout=60.0) as api:
        fleet = (await api.get("/api/map/fleet")).json()
        log("fleet", units=len(fleet["units"]), by_kind={k: sum(u["kind"] == k for u in fleet["units"])
                                                       for k in ("ambulance", "pump", "rescue")},
            routing=fleet["routing"], note="서울 소방서 위치(OSM)에 배치한 시연용 가상 차량")
        turns = []
        for turn, item in enumerate(scenario["turns"], 1):
            turns.append({"speaker": item["speaker"], "text": item["text"]})
            log("turn", turn=turn, speaker="수보요원" if item["speaker"] == 0 else "신고자", text=item["text"],
                prefix_turns=turn)
            fault = decision_faults.get(turn)
            if fault:
                os.environ["SENTINEL_ZORO_URL"] = "http://127.0.0.1:9"
            request_id = f"{log.base['run_id']}-t{turn}"
            t0 = time.perf_counter()
            response = await api.post("/api/compare/decide", json={"turns": turns})
            elapsed = round((time.perf_counter() - t0) * 1000, 1)
            if fault:
                os.environ["SENTINEL_ZORO_URL"] = zoro_url
            if response.status_code != 200:
                log("model_call", turn=turn, request_id=request_id, endpoint="/api/compare/decide -> /v1/stream",
                    status="error", http_status=response.status_code, error=response.json().get("detail"),
                    elapsed_ms=elapsed, injected=bool(fault), fault_note=fault and fault["note"])
                log("policy", turn=turn, action="HOLD", tasks=tasks_view(commits, probs, labels),
                    note="판단 서버 응답 없음: 이전 확정 상태를 유지하고 다음 턴에 전체 대화로 다시 판단")
                continue
            data = response.json()
            labels = data["labels"]
            sft, base = data["results"]["sft"], data["results"]["base"]
            if not model_logged:
                log("model", sft=sft["model"], base=base["model"], profile=data.get("profile"))
                model_logged = True
            probs = sft["probabilities"]
            log("model_call", turn=turn, request_id=request_id, endpoint="/api/compare/decide -> /v1/stream",
                status="ok", input={"turns": turn, "state_chars": data.get("state_chars")},
                output={"sft": {q: rounded(d) for q, d in probs.items()},
                        "base_top": {q: [labels[q].get(top(d)[0], top(d)[0]), round(top(d)[1], 4)]
                                     for q, d in base["probabilities"].items()}},
                latency_ms={"sft": round(sft["latency_ms"], 1), "base": round(base["latency_ms"], 1)},
                elapsed_ms=elapsed, injected=False)
            fresh = advance(commits, probs, turn, labels)
            log("policy", turn=turn, action="COMMIT" if fresh else "WAIT", committed=fresh,
                tasks=tasks_view(commits, probs, labels))

            # Dispatch when the type commits; later EMS commits can upgrade the plan (e.g. 심정지 -> 펌뷸런스).
            if "dispatch" in commits:
                key = tuple(commits.get(t, {}).get("label") for t in ("dispatch", "subtype", "symptom", "urgency"))
                if key != dispatch_key:
                    dispatch_key = key
                    body = {**point, "dispatch": key[0], "subtype": key[1], "symptom": key[2], "urgency": key[3],
                            "busy": [o["unit"]["id"] for o in orders if o["unit"]], "skip": len(orders)}
                    t0 = time.perf_counter()
                    res = await api.post("/api/map/dispatch", json=body)
                    if res.status_code != 200:
                        dispatch_key = None
                        log("tool_call", turn=turn, tool="dispatch_plan", status="error",
                            http_status=res.status_code, note="다음 턴에 다시 시도")
                    else:
                        out = res.json()
                        new = [{"role": a["role"], "kind": a["kind"], "reason": a["reason"], "source": a["source"],
                                "unit": a["unit"] and {k: a["unit"][k] for k in ("id", "name", "station", "district")},
                                "route": a.get("route") and {"source": a["route"]["source"],
                                                             "distance_m": round(a["route"]["distance_m"]),
                                                             "eta_min": round(a["route"]["duration_s"] / 60, 1)}}
                               for a in out["assignments"]]
                        orders += new
                        log("tool_call", turn=turn, tool="dispatch_plan", status="ok",
                            input={k: body[k] for k in ("dispatch", "subtype", "symptom", "urgency", "skip")},
                            plan=[{k: s[k] for k in ("kind", "role")} for s in out["plan"]],
                            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1))
                        for order in new:
                            log("action", turn=turn, action="dispatch_unit", simulated=True, **order)

            # EMS: pick the transfer hospital as soon as 구급 commits; the symptom leader stands in until it commits.
            if commits.get("dispatch", {}).get("key") == "ems":
                if "symptom" in commits:
                    symptom = {"label": commits["symptom"]["label"], "committed": True}
                elif "ems_symptom" in probs:
                    k, p = top(probs["ems_symptom"])
                    symptom = {"label": labels["ems_symptom"].get(k, k), "p": round(p, 4), "committed": False}
                else:
                    symptom = None
                key = (commits.get("subtype", {}).get("label"), symptom and symptom["label"],
                       symptom and symptom["committed"], commits.get("urgency", {}).get("label"))
                if key != transfer_key:
                    transfer_key = key
                    injected = nmc_left > 0
                    body = {**point, "turns": turns, "subtype": key[0], "symptom": key[1], "urgency": key[3]}
                    t0 = time.perf_counter()
                    res = await api.post("/api/map/transfer", json=body)
                    elapsed = round((time.perf_counter() - t0) * 1000, 1)
                    if injected:
                        nmc_left -= 1
                    out = res.json() if res.status_code == 200 else {"status": f"http_{res.status_code}"}
                    if out["status"] != "ok":
                        transfer_key = None  # retry on the next turn
                        log("tool_call", turn=turn, tool="transfer_hospital", status=out["status"], elapsed_ms=elapsed,
                            injected=injected, fault_note=injected and nmc_fault["note"],
                            input={"subtype": key[0], "symptom": symptom, "urgency": key[3]},
                            note="병원 정보를 받지 못함: 병원 추천 보류, 출동은 유지하고 다음 턴에 재조회")
                        log("policy", turn=turn, action="HOLD", task="transfer_hospital",
                            note="병원 후보 없음 (조회 실패) → 보류")
                    else:
                        hospitals = {h["hpid"]: h for h in out["hospitals"]}
                        zoro = out["zoro"]
                        log("tool_call", turn=turn, tool="transfer_hospital", status="ok", data=out["data"],
                            source=out["source"], elapsed_ms=elapsed, injected=False,
                            input={"subtype": key[0], "symptom": symptom, "urgency": key[3]},
                            rule={k: out["rule"].get(k) for k in ("category", "min_level", "prefer", "text", "source")},
                            candidates=[{"hpid": h["hpid"], "name": h["name"], "level": h["level"],
                                         "district": h["district"], "er_beds": h["er_beds"], "beds_at": h["beds_at"],
                                         "eta_min": round(h["route"]["duration_s"] / 60, 1),
                                         "route_source": h["route"]["source"], "data": h["data"],
                                         "option_text": h["context"]} for h in out["hospitals"]])
                        if zoro:
                            log("model_call", turn=turn, request_id=f"{log.base['run_id']}-t{turn}-hospital",
                                endpoint="/v1/choose", status="ok",
                                output={hospitals[k]["name"]: round(v, 4) for k, v in zoro["probabilities"].items()},
                                latency_ms=zoro.get("latency_ms") and round(zoro["latency_ms"], 1))
                            chosen = hospitals[zoro["choice"]]
                            log("action", turn=turn, action="recommend_hospital", simulated=True,
                                hospital=chosen["name"], level=chosen["level"],
                                p=round(zoro["probabilities"][zoro["choice"]], 4),
                                eta_min=round(chosen["route"]["duration_s"] / 60, 1),
                                symptom_committed=symptom and symptom["committed"])
                        else:
                            log("policy", turn=turn, action="HOLD", task="transfer_hospital",
                                note="Zoro 병원 선택 응답 없음 → 가장 가까운 후보를 표시만 하고 확정하지 않음")
    transfer_module._district = real_district

    expected = scenario["expected"]
    task_of = {"dispatch": "dispatch", "ems_subtype": "subtype", "rescue_subtype": "subtype", "urgency": "urgency",
               "ems_symptom": "symptom"}
    check = {q: {"expected": v, "committed": commits.get(task_of[q], {}).get("label"),
                 "turn": commits.get(task_of[q], {}).get("turn"),
                 "match": commits.get(task_of[q], {}).get("label") == v if task_of[q] in commits else None}
             for q, v in expected.items()}
    log("run_end", duration_s=round(time.perf_counter() - started, 2),
        final={t: {k: c[k] for k in ("label", "p", "turn")} for t, c in commits.items()},
        orders=[{"role": o["role"], "unit": o["unit"] and o["unit"]["name"],
                 "eta_min": o["route"] and o["route"]["eta_min"]} for o in orders],
        expected_vs_observed=check,
        note="expected는 시나리오 작성자가 정한 기대 동작이며 모델 입력에 들어가지 않습니다.")
    log.write()
    return log.rows


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    rows = asyncio.run(run(sys.argv[1], sys.argv[2]))
    for row in rows:
        if row["event"] in ("policy", "action", "run_end") or row.get("status") not in (None, "ok"):
            print(json.dumps({k: v for k, v in row.items() if k not in ("run_id", "scenario_id", "at")},
                             ensure_ascii=False)[:400])
