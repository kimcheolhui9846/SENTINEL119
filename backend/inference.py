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


class Inference:
    def decide(self, mode, role, state, criteria, simulation_choices):
        """Return a typed choice envelope and the model ID advertised by Zoro."""
        if role not in _ROLE_TASKS or not criteria or not criteria.keys() <= _ROLE_TASKS[role]:
            raise ValueError("questions must belong to the inference role")
        if mode == "live":
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
