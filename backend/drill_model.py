"""Pinned Zoro adapter for the two synthetic drill scenarios."""

import asyncio
import math

import httpx

from .inference import _CALL_TIMEOUT, InferenceError, _json_request, _role_endpoint
from .team_demo import _model_identity, _turn_prefix


# Final checkpoint H4 (zoro-hier-h4, step 440). Override with ZORO_INTAKE_MODEL_ID when serving another one.
_PINNED_MODEL = (
    "team-demo/sft/zoro-hier-h4@sha256:"
    "2596316b5f53d7cceb99afbc984d21e4ce12b35e2ab2577b7b50cc66b28d264c"
)
_DISPATCH = {"fire", "rescue", "ems", "other"}
_EMS_SUBTYPES = {
    "severe_illness", "illness", "cardiac_arrest", "injury",
    "accident", "pregnancy", "poisoning", "other_ems",
}


def _scores(probabilities, task, labels):
    values = probabilities.get(task)
    if not isinstance(values, dict) or values.keys() != labels:
        raise InferenceError("model_response_invalid")
    scores = values.values()
    if (any(type(score) not in (int, float) or not 0 <= score <= 1 for score in scores)
            or not math.isclose(math.fsum(scores), 1, rel_tol=0, abs_tol=1e-5)):
        raise InferenceError("model_response_invalid")
    return values


async def infer_live(turns, scenario, prior_model):
    url, configured_pin = _role_endpoint("intake")
    payload = {"turns": _turn_prefix({"turns": turns}), "variants": ["sft"]}
    try:
        async with asyncio.timeout(_CALL_TIMEOUT), httpx.AsyncClient(
            timeout=httpx.Timeout(_CALL_TIMEOUT, connect=min(1.0, _CALL_TIMEOUT)),
            follow_redirects=False, trust_env=False, headers={"Accept-Encoding": "identity"},
        ) as client:
            result = await _json_request(client, "POST", url + "/v1/stream", json=payload)
    except (TimeoutError, httpx.TimeoutException):
        raise InferenceError("model_timeout") from None
    except (httpx.HTTPError, httpx.InvalidURL):
        raise InferenceError("model_unavailable") from None

    results = result.get("results")
    sft = results.get("sft") if isinstance(results, dict) else None
    if not isinstance(sft, dict):
        raise InferenceError("model_response_invalid")
    model_id = _model_identity(sft.get("model"))
    if model_id != (configured_pin or _PINNED_MODEL) or (prior_model and model_id != prior_model):
        raise InferenceError("model_identity")
    probabilities = sft.get("probabilities")
    if not isinstance(probabilities, dict):
        raise InferenceError("model_response_invalid")
    dispatch = _scores(probabilities, "dispatch", _DISPATCH)
    subtype = _scores(probabilities, "ems_subtype", _EMS_SUBTYPES) if scenario == "ems" else {}
    return model_id, dispatch, subtype
