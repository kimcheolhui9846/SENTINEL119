"""Role-scoped Zoro transport. The session and Guard own decision validity."""

import asyncio
import json
import os

import httpx


_CALL_TIMEOUT = 5.0
_MAX_BODY_BYTES = 2 * 1024 * 1024
_ROLE_TASKS = {"intake": {"dispatch", "urgency"}, "vehicle": {"vehicle"}, "hospital": {"hospital"}}
_ERROR_MESSAGES = {
    "model_unconfigured": "선택한 역할의 모델 서버 설정이 필요합니다.",
    "model_unavailable": "모델 서버에 연결하지 못했습니다.",
    "model_timeout": "모델 응답 대기 시간을 초과했습니다.",
    "model_identity": "모델 서버에서 사용할 모델 ID를 확인할 수 없습니다.",
    "model_response_invalid": "모델 응답 형식을 확인할 수 없습니다.",
    "fixture_response_invalid": "이 턴의 시뮬레이션 응답을 확인할 수 없습니다.",
}


class InferenceError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(_ERROR_MESSAGES[code])


def _role_endpoint(role):
    prefix = "ZORO_" + role.upper()
    role_url = os.getenv(prefix + "_BASE_URL", "").strip()
    raw_url = role_url or os.getenv("ZORO_BASE_URL", "").strip()
    model_id = os.getenv(prefix + "_MODEL_ID", "").strip()
    if not role_url and not model_id:
        model_id = os.getenv("ZORO_MODEL_ID", "").strip()

    try:
        url = httpx.URL(raw_url)
    except httpx.InvalidURL:
        raise InferenceError("model_unconfigured") from None
    if url.scheme not in {"http", "https"} or not url.host or url.userinfo or url.query or url.fragment:
        raise InferenceError("model_unconfigured")
    return str(url).rstrip("/"), model_id


async def _json_request(client, method, url, **kwargs):
    async with client.stream(method, url, **kwargs) as response:
        response.raise_for_status()
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise InferenceError("model_response_invalid")
        body = bytearray()
        async for chunk in response.aiter_raw():
            if len(body) + len(chunk) > _MAX_BODY_BYTES:
                raise InferenceError("model_response_invalid")
            body.extend(chunk)
    try:
        value = json.loads(body)
    except (ValueError, RecursionError):
        raise InferenceError("model_response_invalid") from None
    if not isinstance(value, dict):
        raise InferenceError("model_response_invalid")
    return value


def _select_model(advertisement, expected):
    data = advertisement.get("data")
    if not isinstance(data, list) or any(
        not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"].strip()
        for entry in data
    ):
        raise InferenceError("model_response_invalid")
    model_ids = {entry["id"] for entry in data}
    if expected and expected in model_ids:
        return expected
    if not expected and len(model_ids) == 1:
        return next(iter(model_ids))
    raise InferenceError("model_identity")


async def _live_decide(role, state, criteria):
    url, expected_model = _role_endpoint(role)
    questions = {task: {"type": "choice", "criteria": choices} for task, choices in criteria.items()}
    try:
        async with asyncio.timeout(_CALL_TIMEOUT):
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(_CALL_TIMEOUT, connect=min(1.0, _CALL_TIMEOUT)),
                follow_redirects=False,
                trust_env=False,
                headers={"Accept-Encoding": "identity"},
            ) as client:
                advertisement = await _json_request(client, "GET", url + "/v1/models")
                model_id = _select_model(advertisement, expected_model)
                result = await _json_request(
                    client,
                    "POST",
                    url + "/v1/systemone",
                    json={"model": model_id, "state": state, "questions": questions, "reasoning": False},
                )
    except (TimeoutError, httpx.TimeoutException):
        raise InferenceError("model_timeout") from None
    except (httpx.HTTPError, httpx.InvalidURL):
        raise InferenceError("model_unavailable") from None

    if "model" in result and result["model"] != model_id:
        raise InferenceError("model_identity")
    answers = result.get("answers")
    if not isinstance(answers, dict):
        raise InferenceError("model_response_invalid")
    return {"answers": {task: answers[task] for task in criteria if task in answers}}, model_id


# SENTINEL_ZORO server (demo/model_server/server.py in the training workspace). It renders the call
# exactly as in training ("화자0/1: text", AI Hub question text and keys), so the console sends turns
# and maps the trained keys back to its Korean labels here.
_SENTINEL_KEYS = {
    "dispatch": {"fire": "화재", "rescue": "구조", "ems": "구급", "other": "기타"},
    "urgency": {"high": "상", "medium": "중", "low": "하"},
}
_HIERARCHY = ("ems_subtype", "rescue_subtype", "ems_symptom")
_CANDIDATE_QUESTIONS = {
    "vehicle": "구급 출동에 보낼 차량을 고르세요. 가용 차량 중 도착 예정시간이 짧은 차량을 우선합니다.",
    "hospital": "환자를 이송할 병원을 고르세요. 수용 가능한 병원 중 도착 예정시간이 짧은 병원을 우선합니다.",
}


def sentinel_url():
    raw = os.getenv("SENTINEL_ZORO_URL", "").strip()
    if not raw:
        return None
    try:
        url = httpx.URL(raw)
    except httpx.InvalidURL:
        raise InferenceError("model_unconfigured") from None
    if url.scheme not in {"http", "https"} or not url.host or url.userinfo or url.query or url.fragment:
        raise InferenceError("model_unconfigured")
    return str(url).rstrip("/")


def _sentinel_turns(turns):
    return [{"speaker": 0 if turn["speaker"] == "수보요원" else 1, "text": turn["text"]}
            for turn in turns if turn["text"].strip()]


def _normalized(probabilities, rename=None):
    values = {(rename or {}).get(key, key): float(value) for key, value in probabilities.items()}
    total = sum(values.values())
    if not values or total <= 0:
        raise InferenceError("model_response_invalid")
    return {key: value / total for key, value in values.items()}


def _choice(probabilities):
    return {"type": "choice", "choice": max(probabilities, key=probabilities.get), "probabilities": probabilities}


async def _sentinel_decide(url, role, state, criteria):
    try:
        async with asyncio.timeout(_CALL_TIMEOUT):
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(_CALL_TIMEOUT, connect=min(1.0, _CALL_TIMEOUT)),
                follow_redirects=False, trust_env=False, headers={"Accept-Encoding": "identity"},
            ) as client:
                turns = _sentinel_turns(state["turns"])
                if not turns:
                    raise InferenceError("model_response_invalid")
                if role == "intake":
                    result = await _json_request(client, "POST", url + "/v1/stream",
                                                 json={"turns": turns, "variants": ["sft"]})
                else:
                    question = {"instructions": _CANDIDATE_QUESTIONS[role], "criteria": criteria[role]}
                    result = await _json_request(client, "POST", url + "/v1/choose",
                                                 json={"turns": turns, "question": question})
    except (TimeoutError, httpx.TimeoutException):
        raise InferenceError("model_timeout") from None
    except (httpx.HTTPError, httpx.InvalidURL):
        raise InferenceError("model_unavailable") from None
    try:
        if role != "intake":
            probabilities = _normalized(result["results"]["probabilities"])
            if set(probabilities) != set(criteria[role]):
                raise InferenceError("model_response_invalid")
            return {"answers": {role: _choice(probabilities)}}, "Zoro-1 base head"
        sft = result["results"]["sft"]
        answers = {task: _choice(_normalized(sft["probabilities"][task], _SENTINEL_KEYS[task]))
                   for task in criteria}
        labels = result.get("labels", {})
        hierarchy = {qid: {"labels": labels.get(qid, {}), "probabilities": _normalized(sft["probabilities"][qid])}
                     for qid in _HIERARCHY if qid in sft["probabilities"]}
        checkpoint = (sft.get("model") or {}).get("checkpoint") or {}
        model_id = "SENTINEL_ZORO" + (f" · {checkpoint['path']}" if checkpoint.get("path") else "")
        return {"answers": answers, "hierarchy": hierarchy}, model_id
    except (KeyError, TypeError, AttributeError, ValueError):
        raise InferenceError("model_response_invalid") from None


class Inference:
    def decide(self, mode, role, state, criteria, simulation_choices):
        """Return a typed choice envelope and the model ID advertised by Zoro."""
        if role not in _ROLE_TASKS or not criteria or not criteria.keys() <= _ROLE_TASKS[role]:
            raise ValueError("questions must belong to the inference role")
        if mode == "live":
            url = sentinel_url()
            if url:
                return asyncio.run(_sentinel_decide(url, role, state, criteria))
            return asyncio.run(_live_decide(role, state, criteria))
        if mode != "simulation":
            raise ValueError("unknown inference mode")

        answers = {}
        for task, choices in criteria.items():
            selected = simulation_choices.get(task)
            if selected not in choices:
                raise InferenceError("fixture_response_invalid")
            answers[task] = {
                "type": "choice",
                "choice": selected,
                "probabilities": {label: float(label == selected) for label in choices},
            }
        return {"answers": answers}, None
