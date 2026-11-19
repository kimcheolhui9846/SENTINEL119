"""Exercise the opt-in console against a fake team-model HTTP transport."""

import asyncio
import json
import os
import time
import unittest
from importlib.util import find_spec
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from backend.inference import InferenceError


CRITERIA = {
    "dispatch": {label: label for label in ("화재", "구조", "구급", "기타")},
    "urgency": {label: label for label in ("상", "중", "하")},
}
TURNS = [{"id": 17, "speaker": "신고자", "text": "도움이 필요해요."}]
MODEL_URL = "http://intake.invalid:8119"
HASHES = {
    "decision_head.pt": "b" * 64,
    "adapter_config.json": "c" * 64,
    "adapter_model.safetensors": "d" * 64,
}


def stream_response():
    return {"profile": "auxiliary", "labels": {}, "state_chars": 20, "results": {"sft": {
        "model": {
            "model_id": "trillionlabs/Zoro-1", "revision": "a" * 40, "variant": "sft",
            "checkpoint": {"path": "zoro-lora-aux-2k", "mode": "lora", "sha256": dict(HASHES), "max_length": 4096},
            "prompt_variant": "bare",
        },
        "probabilities": {
            "dispatch": {"fire": .05, "rescue": .1, "ems": .75, "other": .1},
            "urgency": {"high": .1, "medium": .7, "low": .2},
            "ems_subtype": {"ignored": True},
        },
        "latency_ms": 3,
    }}}


def reply_json(body):
    return httpx.Response(200, stream=httpx.ByteStream(json.dumps(body).encode()))


class TeamDemoTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(find_spec("backend.team_demo"), "opt-in team console adapter is missing")
        from backend.team_demo import TeamDemoInference, create_console

        self.inference = TeamDemoInference()
        self.create_console = create_console
        self.requests = []
        self.response = stream_response()
        self.reply = lambda request: reply_json(self.response)
        env = {key: "" for key in os.environ if key.startswith("ZORO_")}
        env.update({"ZORO_BASE_URL": "http://generic.invalid", "ZORO_INTAKE_BASE_URL": MODEL_URL})
        self.enterContext(patch.dict(os.environ, env))

        client_type = httpx.AsyncClient

        def client_factory(**kwargs):
            return client_type(transport=httpx.MockTransport(self.record_request), **kwargs)

        self.enterContext(patch.object(httpx, "AsyncClient", side_effect=client_factory))

    async def record_request(self, request):
        self.requests.append(request)
        response = self.reply(request)
        return await response if asyncio.iscoroutine(response) else response

    def decide(self, criteria=CRITERIA, *, turns=TURNS):
        return self.inference.decide(
            "live", "intake", {"turns": turns, "future": "미래 발화", "session_id": "private-id"},
            criteria, {"dispatch": "화재", "urgency": "상"},
        )

    def assert_inference_error(self, code, **kwargs):
        with self.assertRaises(InferenceError) as caught:
            self.decide(**kwargs)
        self.assertEqual(caught.exception.code, code)
        for leaked in (MODEL_URL, "private-id", "secret-upstream-body"):
            self.assertNotIn(leaked, str(caught.exception))

    def test_translates_probabilities_and_sends_only_current_turn_fields(self):
        turn = {"id": 18, "speaker": "수보요원", "text": "어디에 계세요?", "evidence": [17]}
        self.decide()
        result, model_id = self.decide(turns=[*TURNS, turn])
        self.assertEqual(result, {"answers": {
            "dispatch": {"type": "choice", "choice": "구급",
                         "probabilities": {"화재": .05, "구조": .1, "구급": .75, "기타": .1}},
            "urgency": {"type": "choice", "choice": "중", "probabilities": {"상": .1, "중": .7, "하": .2}},
        }})
        self.assertRegex(model_id, r"^team-demo/sft/zoro-lora-aux-2k@sha256:[0-9a-f]{64}$")
        for request in self.requests:
            self.assertEqual((request.method, str(request.url)), ("POST", MODEL_URL + "/v1/stream"))
        first = {"speaker": 1, "text": "도움이 필요해요."}
        self.assertEqual([json.loads(request.content) for request in self.requests], [
            {"turns": [first], "variants": ["sft"]},
            {"turns": [first, {"speaker": 0, "text": "어디에 계세요?"}], "variants": ["sft"]},
        ])

    def test_factory_uses_current_prefix_in_live_console_session(self):
        with TestClient(self.create_console()) as client:
            state = client.post("/api/sessions", json={"scenario_id": "reroute", "mode": "live"}).json()
            url = "/api/sessions/" + state["id"]
            self.assertEqual(self.requests, [])
            for _ in range(2):
                response = client.post(url + "/events", json={"action": "next", "expected_version": state["version"]})
                self.assertEqual(response.status_code, 202, response.text)
                state = response.json()
                deadline = time.monotonic() + 3
                while state["processing"] and time.monotonic() < deadline:
                    time.sleep(.01)
                    state = client.get(url).json()
                self.assertFalse(state["processing"])
                self.assertEqual(state["decisions"]["dispatch"]["execution"], "ok")
                self.assertRegex(state["decisions"]["dispatch"]["model_id"], r"^team-demo/sft/")
        bodies = [json.loads(request.content) for request in self.requests if request.url.host == "intake.invalid"]
        turn_one = {"speaker": 1, "text": "뚝섬유원지 북문 앞입니다. 다친 분이 있어서 구급차를 요청해요."}
        turn_two = {"speaker": 0, "text": "현재 위치를 확인했습니다. 차량별 접근 경로가 갱신됐습니다."}
        self.assertEqual(bodies, [
            {"turns": [turn_one], "variants": ["sft"]},
            {"turns": [turn_one, turn_two], "variants": ["sft"]},
        ])

    def test_rejects_invalid_turns_before_network_access(self):
        for turns in (
            [], None, [{"speaker": "상담원", "text": "test"}], [{"speaker": 0, "text": "test"}],
            [{"speaker": True, "text": "test"}], [{"speaker": [], "text": "test"}],
            [{"speaker": "신고자", "text": " \t"}], [{"speaker": "신고자", "text": 5}],
            [{"speaker": "신고자", "text": "x" * 2001}], [{"speaker": "신고자", "text": "\ud800"}],
            TURNS * 121, [None],
        ):
            with self.subTest(turns=repr(turns)[:60]):
                self.assert_inference_error("model_response_invalid", turns=turns)
        self.assertEqual(self.requests, [])

    def test_role_cannot_issue_another_roles_task(self):
        for criteria in ({}, {"vehicle": {"A": "응급차"}}, {"dispatch": CRITERIA["dispatch"], "hospital": {}}):
            with self.subTest(criteria=criteria), self.assertRaises(ValueError):
                self.decide(criteria)
        self.assertEqual(self.requests, [])

    def test_checks_only_requested_task_distributions_and_requires_all_requested(self):
        del self.response["results"]["sft"]["probabilities"]["urgency"]
        result, _ = self.decide({"dispatch": CRITERIA["dispatch"]})
        self.assertEqual(set(result["answers"]), {"dispatch"})
        self.assert_inference_error("model_response_invalid")
        del self.response["results"]["sft"]["probabilities"]["dispatch"]
        self.response["results"]["sft"]["probabilities"]["urgency"] = {"high": .1, "medium": .7, "low": .2}
        result, _ = self.decide({"urgency": CRITERIA["urgency"]})
        self.assertEqual(set(result["answers"]), {"urgency"})

    def test_rejects_malformed_distributions(self):
        for updates in (
            {"ems": True, "fire": 0, "rescue": 0, "other": 0},
            {"ems": float("nan")}, {"ems": float("inf")}, {"ems": .5},
            {"ems": "0.75"}, {"ems": .4, "fire": .4}, {"ems": .76, "fire": -.01},
            {"ems": 100, "fire": -99.2}, {"ems": 10 ** 500},
            {"extra": 0}, {"ems": None}, {"ems": .75002},
        ):
            with self.subTest(updates=updates):
                self.response = stream_response()
                self.response["results"]["sft"]["probabilities"]["dispatch"].update(updates)
                self.assert_inference_error("model_response_invalid")
        self.response = stream_response()
        del self.response["results"]["sft"]["probabilities"]["dispatch"]["fire"]
        self.assert_inference_error("model_response_invalid")
        self.response = stream_response()
        self.response["results"]["sft"]["probabilities"]["dispatch"]["ems"] = .750005
        self.assertEqual(self.decide()[0]["answers"]["dispatch"]["probabilities"]["구급"], .750005)

    def test_pin_covers_all_model_provenance(self):
        _, pinned = self.decide()
        with patch.dict(os.environ, {"ZORO_INTAKE_MODEL_ID": pinned}):
            self.assertEqual(self.decide()[1], pinned)
            for path, value in (
                (("model_id",), "trillionlabs/Zoro-test"), (("revision",), "1" * 40),
                (("checkpoint", "path"), "zoro-new"), (("prompt_variant",), "guided"),
                (("checkpoint", "max_length"), 2048),
                *((("checkpoint", "sha256", name), "1" * 64) for name in HASHES),
            ):
                with self.subTest(path=path):
                    self.response = stream_response()
                    target = self.response["results"]["sft"]["model"]
                    for key in path[:-1]:
                        target = target[key]
                    target[path[-1]] = value
                    self.assert_inference_error("model_identity")

    def test_invalid_checkpoint_provenance_fails_closed(self):
        for model_patch, checkpoint_patch, hashes_patch in (
            ({"model_id": ""}, {}, {}), ({"revision": "a" * 39}, {}, {}), ({"variant": "base"}, {}, {}),
            ({"prompt_variant": []}, {}, {}), ({}, {"path": "../secret-upstream-body"}, {}),
            ({}, {"mode": "head"}, {}), ({}, {"max_length": True}, {}),
            ({}, {}, {"decision_head.pt": "a" * 63}), ({}, {}, {"other": "e" * 64}),
        ):
            with self.subTest(model=model_patch, checkpoint=checkpoint_patch, hashes=hashes_patch):
                self.response = stream_response()
                model = self.response["results"]["sft"]["model"]
                model.update(model_patch)
                model["checkpoint"].update(checkpoint_patch)
                model["checkpoint"]["sha256"].update(hashes_patch)
                self.assert_inference_error("model_identity")
        del self.response["results"]["sft"]["model"]["checkpoint"]["sha256"]["adapter_config.json"]
        self.assert_inference_error("model_identity")

    def test_invalid_upstream_envelopes_and_oversized_responses_are_sanitized(self):
        for body in (
            b"<html>secret-upstream-body</html>", b"[]", b'{"results": []}', b'{"results": {"base": {}}}',
            b'{"results":{"sft":[]}}',
            b'{"pad":"' + b"a" * (2 * 1024 * 1024) + b'"}',
        ):
            with self.subTest(body=body[:32]):
                self.reply = lambda request: httpx.Response(200, stream=httpx.ByteStream(body))
                self.assert_inference_error("model_response_invalid")

    def test_redirect_is_never_followed(self):
        self.reply = lambda request: httpx.Response(307, headers={"Location": "http://redirect.invalid/collect"})
        self.assert_inference_error("model_unavailable")
        self.assertEqual([str(request.url) for request in self.requests], [MODEL_URL + "/v1/stream"])

    def test_total_deadline_cancels_inflight_response(self):
        async def wait_for_response(request):
            await asyncio.sleep(5)
            return reply_json(self.response)

        self.reply = wait_for_response
        with patch("backend.team_demo._CALL_TIMEOUT", .05):
            self.assert_inference_error("model_timeout")
        self.assertEqual(len(self.requests), 1)

    def test_simulation_uses_fixture_and_other_live_roles_use_existing_protocol(self):
        result, model_id = self.inference.decide("simulation", "intake", {}, CRITERIA, {
            "dispatch": "화재", "urgency": "상",
        })
        self.assertIsNone(model_id)
        self.assertEqual(result["answers"]["dispatch"]["choice"], "화재")
        self.assertEqual(self.requests, [])

        for role, choice in (("vehicle", "A"), ("hospital", "H1")):
            with self.subTest(role=role):
                answer = {"type": "choice", "choice": choice, "probabilities": {choice: 1}}
                self.reply = lambda request: reply_json(
                    {"data": [{"id": "generic"}]} if request.method == "GET" else {"answers": {role: answer}},
                )
                result, model_id = self.inference.decide("live", role, {}, {role: {choice: "available"}}, {})
                self.assertEqual((result, model_id), ({"answers": {role: answer}}, "generic"))
        self.assertEqual([(request.url.host, request.url.path) for request in self.requests], [
            ("generic.invalid", "/v1/models"), ("generic.invalid", "/v1/systemone"),
        ] * 2)
