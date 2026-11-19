"""Pinned Zoro adapter for the two synthetic drill scenarios."""

import asyncio

import httpx

from .demo_harness import LABELS, _model_identity, load_profile, valid_distribution
from .inference import _CALL_TIMEOUT, InferenceError, _json_request, _role_endpoint
from .team_demo import _turn_prefix


def _scores(probabilities, task, labels):
    values = probabilities.get(task)
    if not valid_distribution(values, labels):
        raise InferenceError("model_response_invalid")
    return values


async def infer_live(turns, scenario, prior_model):
    url, configured_pin = _role_endpoint("intake")
    try:
        expected_model = configured_pin or load_profile().model_id
    except ValueError:
        raise InferenceError("model_unconfigured") from None
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
    if model_id != expected_model or (prior_model and model_id != prior_model):
        raise InferenceError("model_identity")
    probabilities = sft.get("probabilities")
    if not isinstance(probabilities, dict):
        raise InferenceError("model_response_invalid")
    dispatch = _scores(probabilities, "dispatch", LABELS["dispatch"].keys())
    subtype = _scores(probabilities, "ems_subtype", LABELS["ems_subtype"].keys()) if scenario == "ems" else {}
    return model_id, dispatch, subtype
