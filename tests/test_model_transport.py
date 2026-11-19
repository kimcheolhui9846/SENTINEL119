"""Exercise Zoro requests and failures without a deployed model."""

import asyncio
import json
import os
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.util import find_spec
from unittest.mock import patch

import httpx


INTAKE_CRITERIA = {"dispatch": {"구급": "구급 신고", "화재": "화재 신고"}}
VEHICLE_CRITERIA = {"vehicle": {"V1": "가용 차량"}}


@contextmanager
def model_server(model_ids=("base",), *, response=None, redirect=None, visits=None):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if visits is not None:
                visits.append(self.path)
            if self.path != "/v1/models":
                self.send_error(404)
            elif redirect:
                self.send_response(307)
                self.send_header("Location", redirect)
                self.end_headers()
            else:
                self.send_json({"data": [{"id": model_id} for model_id in model_ids]})

        def do_POST(self):
            if self.path != "/v1/systemone":
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            self.send_json(response)

        def send_json(self, body):
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except ConnectionError:
                pass

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", requests
        finally:
            server.shutdown()
            thread.join()


class ModelTransport(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(find_spec("backend"), "Zoro backend must be importable")
        from backend.inference import Inference, InferenceError
        self.inference = Inference()
        self.error = InferenceError
        # Never select a real endpoint from the shell while exercising the adapter.
        settings = {k: "" for k in os.environ if k.startswith("ZORO_")}
        env = patch.dict(os.environ, settings)
        env.start()
        self.addCleanup(env.stop)

    def test_live_role_uses_explicit_model_and_excludes_fixture_choices(self):
        answer = {"type": "choice", "choice": "구급", "probabilities": {"구급": 1, "화재": 0}}
        with model_server(("base", "intake-v2"), response={"answers": {"dispatch": answer}}) as intake:
            with model_server(("vehicle-v1",), response={"answers": {"vehicle": {
                "type": "choice", "choice": "V1", "probabilities": {"V1": 1},
            }}}) as vehicle:
                with patch.dict(os.environ, {
                    "ZORO_BASE_URL": intake[0],
                    "ZORO_MODEL_ID": "intake-v2",
                    "ZORO_VEHICLE_BASE_URL": vehicle[0],
                }):
                    state = {"turns": [{"id": 1, "text": "도움이 필요합니다."}]}
                    result, model_id = self.inference.decide(
                        "live", "intake", state, INTAKE_CRITERIA, {"dispatch": "화재"},
                    )
                    self.assertEqual(model_id, "intake-v2")
                    self.assertEqual(result, {"answers": {"dispatch": answer}})
                    self.assertEqual(intake[1], [{
                        "model": "intake-v2", "state": state, "questions": {
                            "dispatch": {"type": "choice", "criteria": INTAKE_CRITERIA["dispatch"]},
                        }, "reasoning": False,
                    }])
                    self.assertEqual(self.inference.decide(
                        "live", "vehicle", {}, VEHICLE_CRITERIA, {},
                    )[1], "vehicle-v1")
                    self.assertEqual(vehicle[1][0]["model"], "vehicle-v1")

    def test_unavailable_model_identity_cannot_trigger_inference(self):
        with model_server(("alpha", "beta")) as (url, requests):
            for model_id in ("", "absent"):
                with self.subTest(model=model_id):
                    with patch.dict(os.environ, {
                        "ZORO_BASE_URL": url, "ZORO_MODEL_ID": model_id,
                    }), self.assertRaises(self.error) as caught:
                        self.inference.decide("live", "intake", {}, INTAKE_CRITERIA, {})
                    self.assertEqual(caught.exception.code, "model_identity")
            self.assertEqual(requests, [])

    def test_deadline_includes_both_model_discovery_and_inference(self):
        from backend.inference import _CALL_TIMEOUT, _live_decide

        requests = []
        client_type = httpx.AsyncClient

        async def exercise():
            loop = asyncio.get_running_loop()
            now = loop.time()

            async def reply(request):
                nonlocal now
                requests.append((request.method, request.url.path))
                # Each response fits on its own. The second exceeds the shared budget.
                now += _CALL_TIMEOUT * .6
                # Let the deadline callback run and deliver its cancellation.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                body = {"data": [{"id": "base"}]} if request.method == "GET" else {"answers": {}}
                return httpx.Response(200, stream=httpx.ByteStream(json.dumps(body).encode()))

            def client(**kwargs):
                return client_type(transport=httpx.MockTransport(reply), **kwargs)

            with patch.object(loop, "time", side_effect=lambda: now):
                with patch("backend.inference.httpx.AsyncClient", side_effect=client):
                    with self.assertRaises(self.error) as caught:
                        await _live_decide("intake", {}, INTAKE_CRITERIA)
            self.assertEqual(caught.exception.code, "model_timeout")

        with patch.dict(os.environ, {"ZORO_BASE_URL": "http://127.0.0.1"}):
            asyncio.run(exercise())
        self.assertEqual(requests, [("GET", "/v1/models"), ("POST", "/v1/systemone")])

    def test_invalid_response_is_explicit_and_hides_endpoint(self):
        with model_server(response={"answers": "untyped"}) as (url, requests):
            with patch.dict(os.environ, {"ZORO_BASE_URL": url}):
                with self.assertRaises(self.error) as caught:
                    self.inference.decide("live", "intake", {}, INTAKE_CRITERIA, {})
            self.assertEqual(caught.exception.code, "model_response_invalid")
            self.assertNotIn(url, str(caught.exception))
            self.assertEqual(len(requests), 1)

    def test_large_response_is_rejected(self):
        with model_server(response={"answers": {"unexpected": "x" * 1024}}) as (url, requests):
            with patch.dict(os.environ, {"ZORO_BASE_URL": url}):
                with patch("backend.inference._MAX_BODY_BYTES", 128):
                    with self.assertRaises(self.error) as caught:
                        self.inference.decide("live", "intake", {}, INTAKE_CRITERIA, {})
            self.assertEqual(caught.exception.code, "model_response_invalid")
            self.assertEqual(len(requests), 1)

    def test_unconfigured_live_role_has_no_simulation_fallback(self):
        with self.assertRaises(self.error) as caught:
            self.inference.decide("live", "intake", {}, INTAKE_CRITERIA, {"dispatch": "구급"})
        self.assertEqual(caught.exception.code, "model_unconfigured")

    def test_redirect_does_not_access_another_server(self):
        visits = []
        with model_server(visits=visits) as (target, _):
            with model_server(redirect=f"{target}/v1/models") as (proxy, _):
                with patch.dict(os.environ, {"ZORO_BASE_URL": proxy}):
                    with self.assertRaises(self.error) as caught:
                        self.inference.decide("live", "intake", {}, INTAKE_CRITERIA, {})
            self.assertEqual(caught.exception.code, "model_unavailable")
            self.assertEqual(visits, [])

    def test_simulation_is_identified_by_missing_live_model(self):
        result, model_id = self.inference.decide(
            "simulation", "intake", {}, INTAKE_CRITERIA, {"dispatch": "구급"},
        )
        self.assertIsNone(model_id)
        self.assertEqual(result["answers"]["dispatch"], {
            "type": "choice", "choice": "구급", "probabilities": {"구급": 1.0, "화재": 0.0},
        })

    def test_role_cannot_issue_another_roles_task(self):
        with self.assertRaises(ValueError):
            self.inference.decide("simulation", "intake", {}, VEHICLE_CRITERIA, {"vehicle": "V1"})
