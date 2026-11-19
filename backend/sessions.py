"""Own report state and apply role proposals through the shared decision guard."""

from concurrent.futures import ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from time import monotonic
from uuid import uuid4

from decision_guard import CandidateVersionTracker, Guard

from .inference import Inference, InferenceError
from .scenarios import get_frame, list_scenarios


_INTAKE = ("dispatch", "urgency")
_CANDIDATES = {"vehicle": "vehicles", "hospital": "hospitals"}
_TASKS = (*_INTAKE, *_CANDIDATES)
_RESPONSE_MAX_AGE_S = 5.0
_INTAKE_LABELS = {
    "dispatch": {label: label for label in ("화재", "구조", "구급", "기타")},
    "urgency": {label: label for label in ("상", "중", "하")},
}
_REASONS = {
    "no decision yet": "검토할 정보가 아직 부족합니다.",
    "reviewed support changed": "신고 근거가 갱신되어 판단을 다시 확인합니다.",
    "reviewed location or candidates changed": "위치 또는 후보 정보가 바뀌었습니다.",
    "reviewed candidate version changed": "후보 정보가 갱신되어 다시 평가합니다.",
    "upstream dispatch changed": "출동유형 판단이 바뀌었습니다.",
    "reviewed support matches Zoro choice": "현재 신고 근거와 선택이 일치합니다.",
    "reviewed dispatch, location and candidate match": "확인된 위치와 현재 가용한 후보입니다.",
    "no unique matching reviewed support": "현재 발화에서 판단 근거를 더 확인해야 합니다.",
    "invalid Zoro choice response": "선택 응답의 형식을 확인할 수 없습니다.",
    "Zoro response expired": "모델 응답 시간이 초과되었습니다.",
}


class SessionError(Exception):
    def __init__(self, status, code):
        super().__init__(code)
        self.status = status
        self.code = code


@dataclass
class _Session:
    id: str
    scenario: dict
    mode: str
    guard: Guard
    frame: dict
    lock: RLock = field(default_factory=RLock)
    version: int = 0
    snapshot_version: int = 0
    view_revision: int = 0
    vehicle_recommendation_revision: int = 0
    step: int = 0
    processing: bool = False
    manual: bool = False
    applied_vehicle: str | None = None
    error: str | None = None
    turns: list = field(default_factory=list)
    timeline: list = field(default_factory=list)
    history_count: int = 0
    metadata: dict = field(default_factory=dict)
    eligible: dict = field(default_factory=dict)
    candidate_versions: dict = field(default_factory=dict)
    trackers: dict = field(default_factory=lambda: {
        task: CandidateVersionTracker() for task in _CANDIDATES
    })
    touched_at: float = field(default_factory=monotonic)
    jobs: set = field(default_factory=set)
    latest_job: object | None = None


def _now():
    return datetime.now(timezone.utc)


def _not_expired(candidate):
    expiry = candidate.get("expires_at")
    if expiry is None:
        return True
    try:
        return datetime.fromisoformat(expiry) > _now()
    except (TypeError, ValueError):
        return False


def _available(frame, task):
    if task == "hospital" and frame["external_status"] != "ok":
        return {}
    return {
        item["id"]: item for item in frame[_CANDIDATES[task]]
        if item["availability"] == "available" and _not_expired(item)
    }


def _reason(decision):
    return _REASONS.get(decision.reason, "현재 조건을 확인한 뒤 다시 판단합니다.")


def _wait_reason(session, task, decision):
    if task == "vehicle" and session.applied_vehicle is not None:
        return "모의 출동이 적용되었습니다."
    if session.manual:
        return "직접 입력의 근거가 검토될 때까지 기다립니다."
    if task not in _CANDIDATES:
        return "현재 발화에서 판단 근거를 더 확인해야 합니다."
    if session.guard.active("dispatch").status != "COMMIT":
        return "구급 출동유형 확인을 기다립니다."
    if session.frame["location"] is None:
        return "신고 위치 확인을 기다립니다."
    if task == "hospital":
        source_status = session.frame["external_status"]
        source_message = {
            "unknown": "병원 수용 정보가 아직 확인되지 않았습니다.",
            "timeout": "병원 정보를 조회하지 못했습니다. 병원 상태를 다시 확인할 수 있습니다.",
            "cached": "이전 병원 자료만 남아 있습니다. 최신 수용 정보를 다시 확인할 수 있습니다.",
        }.get(source_status)
        if source_message:
            return source_message
    if not _available(session.frame, task):
        label = "차량" if task == "vehicle" else "병원"
        return f"현재 가용한 {label} 후보가 없습니다."
    if session.processing:
        return "현재 후보의 판단을 기다립니다."
    return _reason(decision)


class SessionService:
    """Run bounded inference jobs outside the lock that serializes report changes."""

    def __init__(self, inference=None):
        self._inference = inference if inference is not None else Inference()
        self._sessions = {}
        self._lock = RLock()
        self._workers = ThreadPoolExecutor(max_workers=8, thread_name_prefix="sentinel-session")
        self._roles = ThreadPoolExecutor(max_workers=8, thread_name_prefix="sentinel-role")

    def close(self):
        self._workers.shutdown(wait=True, cancel_futures=True)
        self._roles.shutdown(wait=True, cancel_futures=True)

    def create(self, scenario_id, mode):
        scenario = next((item for item in list_scenarios() if item["id"] == scenario_id), None)
        if scenario is None:
            raise SessionError(404, "scenario_not_found")
        if mode not in ("simulation", "live"):
            raise SessionError(422, "invalid_mode")
        session_id = str(uuid4())
        session = _Session(
            session_id, scenario, mode, Guard(session_id, max_response_age_s=_RESPONSE_MAX_AGE_S),
            get_frame(scenario_id, 0),
        )
        with self._lock:
            self._prune()
            if len(self._sessions) >= 128:
                raise SessionError(503, "session_capacity")
            self._sessions[session.id] = session
        return self._view(session)

    def get(self, session_id):
        session = self._find(session_id)
        with session.lock:
            return self._view(session)

    def event(self, session_id, action, expected_version, text=None, *, expected_vehicle_id=None,
              expected_vehicle_revision=None):
        session = self._find(session_id)
        with session.lock:
            if type(expected_version) is not int or expected_version != session.version:
                raise SessionError(409, "stale_version")
            if action == "dispatch":
                return self._apply_dispatch(session, expected_vehicle_id, expected_vehicle_revision)
            if action == "next":
                self._next(session)
                tasks = _TASKS
            elif action == "text":
                self._text(session, text)
                tasks = ()
            elif action == "retry":
                tasks = self._retry(session)
            else:
                raise SessionError(422, "invalid_action")

            session.version += 1
            session.snapshot_version += 1
            session.view_revision += 1
            session.error = None
            self._update_guard(session)
            self._queue(session, tasks)
            return self._view(session)

    def _find(self, session_id):
        with self._lock:
            self._prune()
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionError(404, "session_not_found")
            with session.lock:
                session.touched_at = monotonic()
            return session

    def _prune(self):
        cutoff = monotonic() - 30 * 60
        for session_id, session in list(self._sessions.items()):
            with session.lock:
                if session.touched_at < cutoff and not session.jobs:
                    del self._sessions[session_id]

    def _next(self, session):
        if session.manual or session.step >= session.scenario["turn_count"]:
            raise SessionError(409, "replay_finished")
        step = session.step + 1
        frame = get_frame(session.scenario["id"], step)
        if frame["external_status"] in ("timeout", "cached"):
            observations = {}
            for item in session.frame["hospitals"]:
                if item["availability"] != "unknown" and item["source_at"] is not None:
                    observations[item["id"]] = {"availability": item["availability"], "source_at": item["source_at"]}
                elif item.get("last_observation") is not None:
                    observations[item["id"]] = item["last_observation"]
            for item in frame["hospitals"]:
                if item["id"] in observations:
                    item["last_observation"] = observations[item["id"]]
        if session.applied_vehicle is not None:
            self._exclude_applied_vehicle(session, frame)
        session.step, session.frame = step, frame
        session.turns.append({
            "id": len(session.turns) + 1, "speaker": frame["speaker"], "text": frame["text"],
        })

    @staticmethod
    def _exclude_applied_vehicle(session, frame):
        if session.applied_vehicle is None:
            return
        frame["vehicles"] = [
            {**vehicle, "availability": "unavailable", "detail": f'{vehicle["detail"]} · 모의 출동 중'}
            if vehicle["id"] == session.applied_vehicle else vehicle
            for vehicle in frame["vehicles"]
        ]

    @staticmethod
    def _text(session, message):
        if not isinstance(message, str) or not message.strip() or len(message) > 2000:
            raise SessionError(422, "invalid_text")
        if len(session.turns) >= 120:
            raise SessionError(409, "turn_limit")
        session.manual = True
        session.turns.append({
            "id": len(session.turns) + 1, "speaker": "신고자", "text": message.strip(),
        })
        session.frame = {
            **session.frame, "supports": {}, "location": None, "simulation_choices": {},
            "vehicles": [], "hospitals": [], "external_status": "unknown",
        }

    def _retry(self, session):
        if session.manual or session.processing or not session.turns:
            raise SessionError(409, "retry_unavailable")
        if session.frame["external_status"] in ("timeout", "cached"):
            refreshed = get_frame(session.scenario["id"], session.step, hospital_retry=True)
            session.frame = {
                **session.frame, "hospitals": refreshed["hospitals"],
                "external_status": refreshed["external_status"],
                "simulation_choices": {
                    **session.frame["simulation_choices"],
                    **{key: choice for key, choice in refreshed["simulation_choices"].items()
                       if key == "hospital"},
                },
            }
            return ("hospital",)
        failed = tuple(task for task, info in session.metadata.items() if info["execution"] == "error")
        if not failed:
            raise SessionError(409, "retry_unavailable")
        return (*_INTAKE, *_CANDIDATES) if any(task in _INTAKE for task in failed) else failed

    def _apply_dispatch(self, session, expected_vehicle_id, expected_vehicle_revision):
        if session.applied_vehicle is not None:
            raise SessionError(409, "dispatch_already_applied")
        decision = session.guard.active("vehicle")
        if (not isinstance(expected_vehicle_id, str) or expected_vehicle_id != decision.choice
                or type(expected_vehicle_revision) is not int
                or expected_vehicle_revision != session.vehicle_recommendation_revision):
            raise SessionError(409, "stale_vehicle_recommendation")
        if decision.status != "COMMIT" or decision.choice not in _available(session.frame, "vehicle"):
            raise SessionError(409, "vehicle_not_ready")
        session.applied_vehicle = decision.choice
        session.version += 1
        session.view_revision += 1
        self._log(session, len(session.turns), "vehicle", "APPLIED", decision.choice, "모의 출동을 적용했습니다.")
        self._exclude_applied_vehicle(session, session.frame)
        self._update_guard(session)
        return self._view(session)

    def _update_guard(self, session):
        before = {task: session.guard.active(task) for task in _TASKS}
        mode = session.frame["external_status"]
        for task, field_name in _CANDIDATES.items():
            session.eligible[task] = _available(session.frame, task)
            facts = {
                "ranked": {
                    item["id"]: {key: value for key, value in item.items() if key != "source_at"}
                    for item in session.frame[field_name]
                },
                "eligible_ids": sorted(session.eligible[task]),
            }
            if task == "hospital":
                facts["status"] = mode
            session.candidate_versions[task] = session.trackers[task].observe(facts)
        session.guard.advance(
            len(session.turns), supports=session.frame["supports"], location=session.frame["location"],
            vehicles=dict.fromkeys(session.eligible["vehicle"], "available"),
            hospitals=dict.fromkeys(session.eligible["hospital"], "available"),
            candidate_versions=session.candidate_versions,
        )
        self._revise_vehicle_recommendation(session, before["vehicle"])
        for task, previous in before.items():
            if session.guard.active(task) is not previous:
                session.metadata.pop(task, None)
        self._sync_history(session)

    @staticmethod
    def _revise_vehicle_recommendation(session, previous):
        # Guard returns a new decision even when a new result chooses the same vehicle.
        if session.guard.active("vehicle") is not previous:
            session.vehicle_recommendation_revision += 1

    def _queue(self, session, tasks):
        if session.latest_job is not None:
            session.latest_job.cancel()
        session.processing = bool(tasks)
        for task in _TASKS:
            if task in tasks or session.manual or session.metadata.get(task, {}).get("execution") == "running":
                session.metadata.pop(task, None)
        if not tasks:
            return
        future = self._workers.submit(self._run, session, session.snapshot_version, tasks)
        session.latest_job = future
        session.jobs.add(future)
        future.add_done_callback(lambda job: self._reap(session, job))

    @staticmethod
    def _reap(session, job):
        with session.lock:
            session.jobs.discard(job)
            if session.latest_job is job:
                session.latest_job = None

    def _run(self, session, version, tasks):
        try:
            intake = tuple(task for task in _INTAKE if task in tasks)
            if intake:
                self._decide(session, version, "intake", intake)
            with session.lock:
                if session.snapshot_version != version:
                    return
                jobs = [self._roles.submit(self._decide, session, version, role, (role,))
                        for role in _CANDIDATES if role in tasks]
            wait(jobs)
            for job in jobs:
                job.result()
        except Exception:
            with session.lock:
                if session.snapshot_version == version:
                    session.error = "판단 처리 중 오류가 발생했습니다."
        finally:
            with session.lock:
                if session.snapshot_version == version:
                    for item in session.metadata.values():
                        if item["execution"] == "running":
                            item["execution"] = "error"
                    session.processing = False
                    session.view_revision += 1

    def _decide(self, session, version, role, tasks):
        with session.lock:
            if session.snapshot_version != version:
                return
            if role in _CANDIDATES:
                dispatch = session.guard.active("dispatch")
                if (dispatch.status != "COMMIT" or dispatch.choice != "구급"
                        or session.frame["location"] is None or not session.eligible[role]):
                    return
            choices = session.frame["simulation_choices"] if session.mode == "simulation" else {}
            # A fixture with no proposed label represents insufficient information.
            if session.mode == "simulation":
                tasks = tuple(task for task in tasks if task in choices)
            if not tasks:
                return
            if role == "intake":
                criteria = {task: _INTAKE_LABELS[task] for task in tasks}
                state = {"turns": deepcopy(session.turns)}
            else:
                criteria = {role: {item["id"]: item["name"] for item in session.eligible[role].values()}}
                state = {
                    "incident_type": dispatch.choice,
                    "incident_evidence_turns": list(dispatch.evidence_turn_ids),
                    "location": session.frame["location"][0],
                    "location_evidence_turn": session.frame["location"][1],
                    "candidates": deepcopy(list(session.eligible[role].values())),
                    "candidate_snapshot_id": session.candidate_versions[role],
                }
                if session.mode == "simulation" and choices[role] not in criteria[role]:
                    return
            tickets = {task: session.guard.begin(task) for task in tasks}
            for task in tasks:
                session.metadata[task] = {"execution": "running"}
            session.view_revision += 1
            mode = session.mode
            turn_id = len(session.turns)
            applied_vehicle = session.applied_vehicle
            started = monotonic()

        error = None
        model_id = None
        try:
            envelope, model_id = self._inference.decide(mode, role, state, criteria, dict(choices))
        except InferenceError as exc:
            envelope, error = None, str(exc)
        except Exception:
            envelope, error = None, "모델 응답을 처리하지 못했습니다."
        elapsed = round((monotonic() - started) * 1000)
        with session.lock:
            if session.snapshot_version != version:
                for task in tickets:
                    self._log(session, turn_id, task, "STALE", None, "이전 입력 버전의 응답을 폐기했습니다.")
                session.view_revision += 1
                return
            if role in _CANDIDATES and session.candidate_versions[role] != state["candidate_snapshot_id"]:
                session.metadata.pop(role, None)
                self._log(session, turn_id, role, "STALE", None, "이전 후보 목록의 응답을 폐기했습니다.")
                session.view_revision += 1
                return
            if session.applied_vehicle != applied_vehicle:
                # Assignment changes the inventory. Reissue tickets only for roles
                # whose inputs still match, and enforce the original deadline.
                tickets = {task: session.guard.begin(task) for task in tasks}
                if monotonic() - started >= _RESPONSE_MAX_AGE_S:
                    envelope, error = None, "모델 응답 시간이 초과되었습니다."
            if role in _CANDIDATES and _available(session.frame, role) != session.eligible[role]:
                envelope, error = None, "판단 중 후보의 가용 정보가 만료되었습니다."
            previous_vehicle = session.guard.active("vehicle")
            for task, ticket in tickets.items():
                result = session.guard.apply(ticket, envelope)
                if result.status == "STALE":
                    continue
                if result.status == "ERROR" and error is None:
                    error_message = _reason(result)
                else:
                    error_message = error
                if error_message:
                    session.error = error_message
                session.metadata[task] = {
                    "execution": "error" if error_message else "ok",
                    "model_id": model_id if mode == "live" else None,
                    "produced_at": _now().isoformat(),
                    "latency_ms": elapsed,
                    "error": error_message,
                }
            self._revise_vehicle_recommendation(session, previous_vehicle)
            self._sync_history(session)
            session.view_revision += 1

    @staticmethod
    def _log(session, turn, task, status, choice, reason):
        session.timeline.append({
            "id": len(session.timeline) + 1, "turn_id": turn, "task": task,
            "status": status, "choice": choice, "reason": reason,
        })

    def _sync_history(self, session):
        history = session.guard.history
        for item in history[session.history_count:]:
            self._log(session, item.turn_id, item.task, item.status, item.choice, _reason(item))
        session.history_count = len(history)

    def _view(self, session):
        location = session.frame["location"]
        return {
            "id": session.id, "scenario_id": session.scenario["id"],
            "scenario_title": session.scenario["title"], "mode": session.mode,
            "version": session.version, "view_revision": session.view_revision,
            "turn_id": len(session.turns), "total_steps": session.scenario["turn_count"], "step": session.step,
            "processing": session.processing, "manual": session.manual, "turns": deepcopy(session.turns),
            "decisions": {task: self._decision_view(session, task) for task in _TASKS},
            "vehicles": deepcopy(session.frame["vehicles"]),
            "hospitals": deepcopy(session.frame["hospitals"]),
            "location": location[0] if location else None, "external_status": session.frame["external_status"],
            "history": deepcopy(session.timeline), "applied_vehicle": session.applied_vehicle,
            "error": session.error,
        }

    @staticmethod
    def _decision_view(session, task):
        result = session.guard.active(task)
        info = session.metadata.get(task, {})
        waiting_for_information = result.status == "WAIT" and (
            task != "vehicle" or session.applied_vehicle is None
        )
        reason = info.get("error")
        if reason is None:
            if result.status == "OUT_OF_SCOPE":
                reason = "구급 유형에서 제공하는 판단입니다."
            elif result.status == "WAIT":
                reason = _wait_reason(session, task, result)
            else:
                reason = _reason(result)
        return {
            "status": result.status, "choice": result.choice, "reason": reason,
            "evidence_turn_ids": list(result.evidence_turn_ids),
            "probabilities": (dict(result.probabilities)
                              if result.probabilities is not None and session.mode == "live" else None),
            "execution": info.get("execution", "idle"),
            "model_id": info.get("model_id"), "guideline_refs": [],
            "missing_information": [reason] if waiting_for_information else [],
            "candidate_snapshot_id": session.candidate_versions.get(task),
            **({"recommendation_revision": session.vehicle_recommendation_revision} if task == "vehicle" else {}),
            "produced_at": info.get("produced_at"), "latency_ms": info.get("latency_ms"),
        }
