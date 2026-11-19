"""Validated runtime policy for the demo model's turn-wise predictions."""

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from src.sentinel119.hierarchy import QUESTIONS as HIERARCHY_QUESTIONS

from .inference import InferenceError, _SENTINEL_KEYS


_DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "demo-harness.json"
_PROFILE_FIELDS = {"version", "model_id", "thresholds", "dispatch_overrides", "withdrawals"}
_THRESHOLD_FIELDS = {"dispatch", "subtype", "urgency", "symptom"}
_CHECKPOINT_FILES = {"decision_head.pt", "adapter_config.json", "adapter_model.safetensors"}
_SLUG = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}"
LABELS = {**_SENTINEL_KEYS, **{task: question["criteria"] for task, question in HIERARCHY_QUESTIONS.items()}}
_THRESHOLD_BY_HEAD = {
    "dispatch": "dispatch", "urgency": "urgency", "ems_subtype": "subtype",
    "rescue_subtype": "subtype", "ems_symptom": "symptom",
}
_SUBTYPE_HEADS = {head for head, task in _THRESHOLD_BY_HEAD.items() if task == "subtype"}


@dataclass(frozen=True, slots=True)
class DemoHarnessProfile:
    version: str
    model_id: str
    policy_digest: str
    thresholds: Mapping
    dispatch_overrides: Mapping
    subtype_overrides: Mapping
    withdrawals: frozenset[str]


def _matches(pattern, value):
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def _model_identity(model):
    if not isinstance(model, dict) or model.get("variant") != "sft":
        raise InferenceError("model_identity")
    model_id, revision, prompt = model.get("model_id"), model.get("revision"), model.get("prompt_variant")
    if (not _matches(rf"{_SLUG}(?:/{_SLUG})?", model_id)
            or not _matches(r"[0-9a-fA-F]{40}", revision)
            or not _matches(_SLUG, prompt)):
        raise InferenceError("model_identity")

    checkpoint = model.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("mode") != "lora":
        raise InferenceError("model_identity")
    name, digests = checkpoint.get("path"), checkpoint.get("sha256")
    if (not _matches(_SLUG, name) or not isinstance(digests, dict) or digests.keys() != _CHECKPOINT_FILES
            or any(not _matches(r"[0-9a-fA-F]{64}", value) for value in digests.values())):
        raise InferenceError("model_identity")

    identity = {
        "model_id": model_id, "revision": revision.lower(), "variant": "sft", "prompt_variant": prompt,
        "checkpoint": {"path": name, "mode": "lora", "sha256": {key: value.lower() for key, value in digests.items()}},
    }
    if "max_length" in checkpoint:
        max_length = checkpoint["max_length"]
        if type(max_length) is not int or max_length <= 0:
            raise InferenceError("model_identity")
        identity["checkpoint"]["max_length"] = max_length
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"team-demo/sft/{name}@sha256:{hashlib.sha256(encoded).hexdigest()}"


def _unique_object(pairs):
    data = {}
    for key, value in pairs:
        if key in data:
            raise ValueError("duplicate JSON key")
        data[key] = value
    return data


def _finite_float(literal):
    value = float(literal)
    if not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    return value


def parse_json(raw):
    """Reject ambiguous or nonfinite JSON before it enters a relay or profile."""
    return json.loads(raw, object_pairs_hook=_unique_object, parse_float=_finite_float, parse_constant=_finite_float)


def _normalized(value):
    return type(value) in (int, float) and 0 <= value <= 1


def _nonblank(value):
    return isinstance(value, str) and bool(value.strip())


def load_profile(path=None):
    """Read a frozen profile, with an optional configured path for the demo runtime."""
    source = path if path is not None else os.getenv("SENTINEL_DEMO_HARNESS_PROFILE") or _DEFAULT_PATH
    try:
        profile = parse_json(Path(source).read_text(encoding="utf-8"))
        if not isinstance(profile, dict) or profile.keys() - {"subtype_overrides"} != _PROFILE_FIELDS:
            raise ValueError("invalid profile fields")
        version, model_id = profile["version"], profile["model_id"]
        if not _nonblank(version) or not _matches(rf"team-demo/sft/{_SLUG}@sha256:[0-9a-f]{{64}}", model_id):
            raise ValueError("invalid profile identity")
        thresholds, dispatch_overrides = profile["thresholds"], profile["dispatch_overrides"]
        if (not isinstance(thresholds, dict) or thresholds.keys() != _THRESHOLD_FIELDS
                or not all(map(_normalized, thresholds.values()))):
            raise ValueError("invalid profile thresholds")
        if (not isinstance(dispatch_overrides, dict) or not dispatch_overrides.keys() <= _SENTINEL_KEYS["dispatch"].keys()
                or not all(map(_normalized, dispatch_overrides.values()))):
            raise ValueError("invalid dispatch overrides")
        subtype_overrides = profile.get("subtype_overrides", {})
        if (not isinstance(subtype_overrides, dict) or not subtype_overrides.keys() <= _SUBTYPE_HEADS
                or not all(map(_normalized, subtype_overrides.values()))):
            raise ValueError("invalid subtype overrides")
        withdrawals = profile["withdrawals"]
        if (not isinstance(withdrawals, list) or not all(map(_nonblank, withdrawals))
                or len(set(withdrawals)) != len(withdrawals)):
            raise ValueError("invalid profile withdrawals")
        canonical = json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except (OSError, TypeError, ValueError, RecursionError) as exc:
        raise ValueError("demo harness profile is invalid") from exc
    return DemoHarnessProfile(version, model_id, digest, MappingProxyType(thresholds), MappingProxyType(dispatch_overrides),
                              MappingProxyType(subtype_overrides), frozenset(withdrawals))


def valid_distribution(scores, expected_keys):
    return (
        isinstance(scores, dict) and scores.keys() == expected_keys
        and all(map(_normalized, scores.values()))
        and math.isclose(math.fsum(scores.values()), 1, rel_tol=0, abs_tol=1e-5)
    )


def _model(head):
    model = head.get("model") if isinstance(head, dict) else None
    if not isinstance(model, dict):
        raise InferenceError("model_response_invalid")
    return model


def _validate_models(results, expected):
    sft = _model(results.get("sft"))
    base = _model(results.get("base"))
    revision = base.get("revision")
    if (_model_identity(sft) != expected or base.get("variant") != "base" or base.get("checkpoint") is not None
            or base.get("model_id") != sft["model_id"] or not isinstance(revision, str)
            or revision.lower() != sft["revision"].lower()):
        raise InferenceError("model_identity")


def _validate_labels(labels):
    if (not isinstance(labels, dict) or not _SENTINEL_KEYS.keys() <= labels.keys()
            or not labels.keys() <= LABELS.keys() & _THRESHOLD_BY_HEAD.keys()
            or any(choices != LABELS[task] for task, choices in labels.items())):
        raise InferenceError("model_response_invalid")


def _gate(turns, withdrawals):
    if not isinstance(turns, (list, tuple)) or not 1 <= len(turns) <= 120 or any(
        not isinstance(turn, dict) or type(turn.get("speaker")) is not int or turn["speaker"] not in (0, 1)
        or not _nonblank(turn.get("text")) or len(turn["text"]) > 2000 for turn in turns
    ):
        raise InferenceError("model_response_invalid")
    if any(turn["speaker"] == 1 and turn["text"] in withdrawals for turn in turns):
        return "review"
    return "allow" if turns[-1]["speaker"] == 1 else "wait"


def _choices(probabilities, labels, profile, gate):
    if (not isinstance(probabilities, dict) or not _SENTINEL_KEYS.keys() <= probabilities.keys()
            or not probabilities.keys() <= labels.keys()):
        raise InferenceError("model_response_invalid")
    choices = {}
    for task, scores in probabilities.items():
        if not valid_distribution(scores, labels[task].keys()):
            raise InferenceError("model_response_invalid")
        if gate != "allow":
            continue
        key = max(scores, key=scores.get)
        probability = scores[key]
        threshold = profile.thresholds[_THRESHOLD_BY_HEAD[task]]
        if task == "dispatch":
            threshold = profile.dispatch_overrides.get(key, threshold)
        else:
            threshold = profile.subtype_overrides.get(task, threshold)
        if probability >= threshold and sum(p == probability for p in scores.values()) == 1:
            choices[task] = {"key": key, "label": labels[task][key], "p": probability}
    return choices


def apply_harness(turns, response, profile):
    """Validate paired model results and emit only choices allowed by the active policy."""
    gate = _gate(turns, profile.withdrawals)
    results = response.get("results") if isinstance(response, dict) else None
    if not isinstance(results, dict):
        raise InferenceError("model_response_invalid")
    _validate_models(results, profile.model_id)
    labels = response.get("labels")
    _validate_labels(labels)
    choices = {variant: _choices(results[variant].get("probabilities"), labels, profile, gate)
               for variant in ("sft", "base")}
    return {"model_id": profile.model_id, "policy_digest": profile.policy_digest, "thresholds": dict(profile.thresholds),
            "dispatch_overrides": dict(profile.dispatch_overrides), "subtype_overrides": dict(profile.subtype_overrides),
            "gate": gate, "choices": choices}
