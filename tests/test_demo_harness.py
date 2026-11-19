import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import compare, drill_model


ROOT = Path(__file__).resolve().parents[1]
CALLER = {"speaker": 1, "text": "합성 신고입니다."}
OPERATOR = {"speaker": 0, "text": "말씀해 주세요."}
WITHDRAWAL = "신고를 취소합니다."
LABELS = {"dispatch": {"fire": "화재", "rescue": "구조", "ems": "구급", "other": "기타"},
          "urgency": {"high": "상", "medium": "중", "low": "하"}}
RESCUE_LABELS = {"safety": "안전사고", "property": "대물사고", "suicide": "자살", "other_rescue": "기타구조"}
MODEL = {
    "model_id": "test/Zoro-1", "revision": "a" * 40, "variant": "sft", "prompt_variant": "bare",
    "checkpoint": {
        "path": "rotated-lora", "mode": "lora", "max_length": 4096,
        "sha256": {"decision_head.pt": "5" * 64, "adapter_config.json": "6" * 64,
                   "adapter_model.safetensors": "7" * 64},
    },
}
FINGERPRINT = "team-demo/sft/rotated-lora@sha256:" + hashlib.sha256(
    json.dumps(MODEL, sort_keys=True, separators=(",", ":")).encode(),
).hexdigest()
PROFILE = {"version": "demo-test-v1", "model_id": FINGERPRINT,
           "thresholds": {"dispatch": .90, "subtype": .85, "urgency": .85, "symptom": .80},
           "dispatch_overrides": {"fire": .75}, "withdrawals": [WITHDRAWAL]}


def response():
    return {
        "profile": "hierarchy", "labels": {task: labels.copy() for task, labels in LABELS.items()},
        "state_chars": 15,
        "results": {
            "sft": {"model": MODEL, "latency_ms": 1, "probabilities": {
                "dispatch": {"fire": .75, "rescue": .10, "ems": .1, "other": .05},
                "urgency": {"high": .85, "medium": .1, "low": .05},
            }},
            "base": {"model": {**MODEL, "variant": "base", "checkpoint": None}, "latency_ms": 2,
                     "probabilities": {"dispatch": {"fire": .10, "rescue": .01, "ems": .89, "other": 0},
                                       "urgency": {"high": .50, "medium": .50, "low": 0}}},
        },
    }


class DemoHarnessTests(unittest.TestCase):
    def write(self, path_payload):
        with tempfile.NamedTemporaryFile(dir=ROOT, suffix=".json", delete=False) as stream:
            path = Path(stream.name)
            self.addCleanup(path.unlink)
            stream.write(path_payload)
        return path

    def profile_path(self, **overrides):
        return self.write(json.dumps({**PROFILE, **overrides}, ensure_ascii=False).encode("utf-8"))

    def harness_module(self):
        self.assertTrue(hasattr(compare, "apply_harness"), "compare must use the shared demo harness")
        from backend import demo_harness
        return demo_harness

    def post(self, payload, profile_path, *, turns=None, status=200, raw=None):
        app = FastAPI()
        app.include_router(compare.router)
        if raw is None:
            upstream = httpx.Response(status, json=payload)
        else:
            upstream = httpx.Response(status, content=raw)
        with TestClient(app, raise_server_exceptions=False) as client, patch.dict("os.environ", {
            "SENTINEL_ZORO_URL": "http://localhost:8001", "SENTINEL_DEMO_HARNESS_PROFILE": str(profile_path),
        }), patch("backend.compare.httpx.post", return_value=upstream) as relay:
            result = client.post("/api/compare/decide", json={"turns": turns or [CALLER]})
        return result, relay

    def test_custom_profile_accepts_rotated_lora_and_keeps_raw_fields(self):
        result, relay = self.post(response(), self.profile_path())
        self.assertEqual(result.status_code, 200)
        value = result.json()
        self.assertIn("harness", value)
        harness = value.pop("harness")
        self.assertEqual(value, response())
        self.assertEqual(harness["model_id"], FINGERPRINT)
        self.assertEqual(harness["thresholds"], PROFILE["thresholds"])
        self.assertEqual(harness["dispatch_overrides"], {"fire": .75})
        self.assertEqual(harness["subtype_overrides"], {})
        self.assertEqual(harness["gate"], "allow")
        self.assertEqual(harness["choices"], {"sft": {
            "dispatch": {"key": "fire", "label": "화재", "p": .75},
            "urgency": {"key": "high", "label": "상", "p": .85},
        }, "base": {}})
        self.assertEqual(relay.call_args.kwargs["json"], {"turns": [CALLER], "variants": ["sft", "base"]})

    def test_checkpoint_or_base_identity_mismatch_is_sanitized(self):
        for changed_base in ({"revision": "b" * 40}, {"checkpoint": MODEL["checkpoint"]}, {"variant": "sft"}):
            data = response()
            data["results"]["base"]["model"].update(changed_base)
            with self.subTest(changed_base=changed_base):
                result, _ = self.post(data, self.profile_path())
                self.assertEqual((result.status_code, result.json()), (502, {"detail": "compare_unavailable"}))
        path = self.profile_path(model_id="team-demo/sft/other@sha256:" + "2" * 64)
        result, _ = self.post(response(), path)
        self.assertEqual((result.status_code, result.json()), (502, {"detail": "compare_unavailable"}))

    def test_incomplete_or_invalid_scores_and_labels_fail_closed(self):
        path = self.profile_path()
        for invalid in (
            {"fire": 1}, {"fire": True, "rescue": 0, "ems": 0, "other": 0},
            {"fire": .8, "rescue": .2, "ems": .1, "other": 0},
            {"fire": 1.1, "rescue": -.1, "ems": 0, "other": 0},
        ):
            with self.subTest(invalid=invalid):
                data = response()
                data["results"]["sft"]["probabilities"]["dispatch"] = invalid
                result, _ = self.post(data, path)
                self.assertEqual(result.status_code, 502)
        data = response()
        del data["results"]["sft"]["probabilities"]["urgency"]
        self.assertEqual(self.post(data, path)[0].status_code, 502)
        data = response()
        data["labels"]["dispatch"]["fire"] = "구급"
        self.assertEqual(self.post(data, path)[0].status_code, 502)

    def test_malformed_and_duplicate_upstream_json_fail_closed(self):
        path = self.profile_path()
        original = json.dumps(response(), ensure_ascii=False).encode("utf-8")
        for raw in (b"invalid", original.replace(b'"fire": 0.75', b'"fire": 0.75, "fire": 0.75'),
                    original.replace(b'"fire": 0.75', b'"fire": NaN'), original.replace(b'"latency_ms": 1', b'"latency_ms": 1e999')):
            with self.subTest(raw=raw[:20]):
                result, _ = self.post(None, path, raw=raw)
                self.assertEqual((result.status_code, result.json()), (502, {"detail": "compare_unavailable"}))

    def test_optional_heads_require_complete_known_mappings_when_present(self):
        path = self.profile_path()
        data = response()
        data["labels"]["rescue_subtype"] = RESCUE_LABELS.copy()
        data["results"]["sft"]["probabilities"]["rescue_subtype"] = {
            "safety": .85, "property": .10, "suicide": 0, "other_rescue": .05,
        }
        result, _ = self.post(data, path)
        self.assertEqual(result.json()["harness"]["choices"]["sft"]["rescue_subtype"],
                         {"key": "safety", "label": "안전사고", "p": .85})
        for labels in ({**RESCUE_LABELS, "safety": "대물사고", "property": "안전사고"},
                       {key: label for key, label in RESCUE_LABELS.items() if key != "suicide"}):
            with self.subTest(labels=labels):
                data["labels"]["rescue_subtype"] = labels
                result, _ = self.post(data, path)
                self.assertEqual((result.status_code, result.json()), (502, {"detail": "compare_unavailable"}))

    def test_subtype_override_controls_the_named_head(self):
        path = self.profile_path(subtype_overrides={"rescue_subtype": .68})
        data = response()
        data["labels"]["rescue_subtype"] = RESCUE_LABELS.copy()
        scores = {"safety": .68, "property": .25, "suicide": 0, "other_rescue": .07}
        data["results"]["sft"]["probabilities"]["rescue_subtype"] = scores
        result, _ = self.post(data, path)
        self.assertEqual(result.status_code, 200)
        harness = result.json()["harness"]
        self.assertEqual(harness["subtype_overrides"], {"rescue_subtype": .68})
        self.assertEqual(harness["choices"]["sft"]["rescue_subtype"], {"key": "safety", "label": "안전사고", "p": .68})
        legacy, _ = self.post(data, self.profile_path())
        self.assertNotIn("rescue_subtype", legacy.json()["harness"]["choices"]["sft"])
        self.assertNotEqual(harness["policy_digest"], legacy.json()["harness"]["policy_digest"])

    def test_default_profile_matches_published_h4_dev83_thresholds(self):
        profile = self.harness_module().load_profile(ROOT / "backend/data/demo-harness.json")
        self.assertEqual(profile.version, "h4-dev83-v1")
        self.assertEqual(profile.thresholds, {"dispatch": .82, "subtype": .86, "urgency": .82, "symptom": .69})
        self.assertEqual(profile.dispatch_overrides, {"ems": .66, "rescue": .86, "fire": .79, "other": .82})
        self.assertEqual(profile.subtype_overrides, {"ems_subtype": .86, "rescue_subtype": .68})

    def test_taxonomy_head_without_policy_threshold_is_sanitized(self):
        task = "future_head"
        labels = {"first": "첫째", "second": "둘째"}
        data = response()
        data["labels"][task] = labels
        data["results"]["sft"]["probabilities"][task] = {"first": 1, "second": 0}
        with patch.dict(self.harness_module().LABELS, {task: labels}):
            result, _ = self.post(data, self.profile_path())
        self.assertEqual(result.status_code, 502)
        self.assertEqual(result.json(), {"detail": "compare_unavailable"})

    def test_tied_head_never_has_choice_even_with_low_threshold(self):
        path = self.profile_path(thresholds=dict.fromkeys(PROFILE["thresholds"], 0), dispatch_overrides={})
        data = response()
        data["results"]["sft"]["probabilities"]["dispatch"] = {"fire": .5, "rescue": 0, "ems": .5, "other": 0}
        result, _ = self.post(data, path)
        harness = result.json()["harness"]
        self.assertNotIn("dispatch", harness["choices"]["sft"])
        self.assertNotIn("urgency", harness["choices"]["base"])

    def test_withdrawal_latches_review_until_a_new_prefix(self):
        run = self.harness_module()
        profile = run.load_profile(self.profile_path())
        for prefix, gate in (([CALLER, OPERATOR], "wait"),
                             ([CALLER, {**CALLER, "text": WITHDRAWAL}, OPERATOR, CALLER], "review"),
                             ([CALLER, {**CALLER, "text": WITHDRAWAL}, OPERATOR], "review")):
            harness = run.apply_harness(prefix, response(), profile)
            self.assertEqual(harness["gate"], gate)
            self.assertEqual(harness["choices"], {"sft": {}, "base": {}})
        quote = {**OPERATOR, "text": WITHDRAWAL}
        for prefix in ([quote, CALLER], [{**CALLER, "text": f"{WITHDRAWAL} "}], [CALLER]):
            self.assertEqual(run.apply_harness(prefix, response(), profile)["gate"], "allow")

    def test_policy_digest_is_canonical_and_invalid_profile_never_relays(self):
        run = self.harness_module()
        path = self.profile_path()
        raw = json.dumps(PROFILE, indent=2, sort_keys=True).encode("utf-8")
        self.assertEqual(run.load_profile(path).policy_digest, run.load_profile(self.write(raw)).policy_digest)
        malformed = (
            self.profile_path(model_id="invalid"), self.profile_path(thresholds={}),
            self.profile_path(dispatch_overrides={"fire": True}), self.profile_path(dispatch_overrides={"unseen": .75}),
            self.profile_path(subtype_overrides={"rescue_subtype": True}), self.profile_path(subtype_overrides={"urgency": .68}),
            self.profile_path(subtype_overrides={"ems_symptom": .69}), self.profile_path(subtype_overrides=None),
            self.profile_path(withdrawals=[WITHDRAWAL, WITHDRAWAL]),
            self.write(raw.replace(b'"fire": 0.75', b'"fire": 0.75, "fire": 0.75')),
        )
        for path in malformed:
            with self.subTest(path=path.name):
                result, relay = self.post(response(), path)
                self.assertEqual((result.status_code, result.json()), (503, {"detail": "compare_unconfigured"}))
                relay.assert_not_called()

    def test_drill_uses_runtime_profile_and_rejects_prior_checkpoint_change(self):
        path = self.profile_path()
        prefix = [{"speaker": "신고자", "text": CALLER["text"]}]
        model_response = {"results": {"sft": response()["results"]["sft"]}}
        with patch.dict("os.environ", {"ZORO_INTAKE_BASE_URL": "http://localhost:8001", "ZORO_INTAKE_MODEL_ID": "",
                                      "SENTINEL_DEMO_HARNESS_PROFILE": str(path)}), patch(
            "backend.drill_model._json_request", new_callable=AsyncMock, return_value=model_response,
        ):
            model_id, scores, _ = asyncio.run(drill_model.infer_live(prefix, "fire", None))
            self.assertEqual((model_id, scores["fire"]), (FINGERPRINT, .75))
            with self.assertRaisesRegex(drill_model.InferenceError, "ID"):
                asyncio.run(drill_model.infer_live(prefix, "fire", "another-checkpoint"))
