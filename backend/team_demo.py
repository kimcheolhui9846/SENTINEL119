"""Opt-in console adapter for the team SFT demo's turn-wise intake endpoint."""

import asyncio

import httpx

from .api import create_app
from .demo_harness import _model_identity, valid_distribution
from .inference import _CALL_TIMEOUT, Inference, InferenceError, _json_request, _role_endpoint, sentinel_url


_SPEAKERS = {"수보요원": 0, "신고자": 1}
_LABELS = {
    "dispatch": {"fire": "화재", "rescue": "구조", "ems": "구급", "other": "기타"},
    "urgency": {"high": "상", "medium": "중", "low": "하"},
}


def _turn_prefix(state):
    turns = state.get("turns") if isinstance(state, dict) else None
    if not isinstance(turns, list) or not 1 <= len(turns) <= 120:
        raise InferenceError("model_response_invalid")
    prefix = []
    for turn in turns:
        if not isinstance(turn, dict):
            raise InferenceError("model_response_invalid")
        speaker, text = turn.get("speaker"), turn.get("text")
        if not isinstance(speaker, str) or speaker not in _SPEAKERS:
            raise InferenceError("model_response_invalid")
        if not isinstance(text, str) or len(text) > 2000 or not text.strip():
            raise InferenceError("model_response_invalid")
        prefix.append({"speaker": _SPEAKERS[speaker], "text": text})
    return prefix


def _choice_answer(distribution, labels):
    if not valid_distribution(distribution, labels.keys()):
        raise InferenceError("model_response_invalid")
    scores = list(distribution.values())
    winner = max(distribution, key=distribution.get)
    if scores.count(distribution[winner]) != 1:
        raise InferenceError("model_response_invalid")
    return {
        "type": "choice",
        "choice": labels[winner],
        "probabilities": {label: distribution[key] for key, label in labels.items()},
    }


async def _live_intake(state, criteria):
    url, expected_model = _role_endpoint("intake")
    payload = {"turns": _turn_prefix(state), "variants": ["sft"]}
    try:
        async with asyncio.timeout(_CALL_TIMEOUT):
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(_CALL_TIMEOUT, connect=min(1.0, _CALL_TIMEOUT)),
                follow_redirects=False,
                trust_env=False,
                headers={"Accept-Encoding": "identity"},
            ) as client:
                result = await _json_request(client, "POST", url + "/v1/stream", json=payload)
    except (TimeoutError, httpx.TimeoutException):
        raise InferenceError("model_timeout") from None
    except (httpx.HTTPError, httpx.InvalidURL):
        raise InferenceError("model_unavailable") from None
    except UnicodeEncodeError:
        raise InferenceError("model_response_invalid") from None

    variants = result.get("results")
    sft = variants.get("sft") if isinstance(variants, dict) else None
    if not isinstance(sft, dict):
        raise InferenceError("model_response_invalid")
    model_id = _model_identity(sft.get("model"))
    if expected_model and model_id != expected_model:
        raise InferenceError("model_identity")
    probabilities = sft.get("probabilities")
    if not isinstance(probabilities, dict):
        raise InferenceError("model_response_invalid")
    answers = {task: _choice_answer(probabilities.get(task), _LABELS[task]) for task in criteria}
    return {"answers": answers}, model_id


class TeamDemoInference(Inference):
    def decide(self, mode, role, state, criteria, simulation_choices):
        # With SENTINEL_ZORO_URL the base adapter also returns subtype/symptom for the confidence rule.
        if mode != "live" or role != "intake" or sentinel_url():
            return super().decide(mode, role, state, criteria, simulation_choices)
        if not criteria or not criteria.keys() <= _LABELS.keys():
            raise ValueError("questions must belong to the inference role")
        if any(choices.keys() != set(_LABELS[task].values()) for task, choices in criteria.items()):
            raise ValueError("intake questions must use the console's label sets")
        return asyncio.run(_live_intake(state, criteria))


def create_console():
    return create_app(inference=TeamDemoInference())
