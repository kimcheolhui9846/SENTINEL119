"""Public contract for the local dispatch demo. Fixtures never need real model weights."""
import time
import unittest
from threading import Event
from importlib.util import find_spec


class ConsoleFlow(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(find_spec("backend"), "FastAPI console backend must be importable")
        from fastapi.testclient import TestClient
        from backend.api import create_app
        self.client = TestClient(create_app())
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def create(self, scenario="staged", mode="simulation"):
        response = self.client.post("/api/sessions", json={"scenario_id":scenario, "mode":mode})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def event(self, state, action="next", **fields):
        url = "/api/sessions/" + state["id"]
        response = self.client.post(url + "/events", json={
            "action":action, "expected_version":state["version"], **fields})
        self.assertEqual(response.status_code, 202, response.text)
        end = time.monotonic() + 5
        result = response.json()
        while result["processing"] and time.monotonic() < end:
            result = self.client.get(url).json()
            time.sleep(.01)
        self.assertFalse(result["processing"])
        return result

    def test_three_replays_and_stale_writes(self):
        scenarios = self.client.get("/api/scenarios").json()["scenarios"]
        self.assertEqual({s["id"] for s in scenarios}, {"staged","reroute","outage"})
        for scenario in scenarios:
            with self.subTest(scenario=scenario["id"]):
                state = self.create(scenario["id"])
                self.assertEqual(state["turns"], [])
                self.assertEqual(set(state["decisions"]), {"dispatch","urgency","vehicle","hospital"})
                for _ in range(scenario["turn_count"]):
                    old = state
                    state = self.event(state)
                    self.assertEqual(len(state["turns"]), old["turn_id"] + 1)
                    retry = self.client.post("/api/sessions/"+state["id"]+"/events",
                        json={"action":"next","expected_version":old["version"]})
                    self.assertEqual(retry.status_code, 409)
                self.assertEqual(state["decisions"]["vehicle"]["status"], "COMMIT")

    def test_manual_input_withdraws_earlier_evidence(self):
        state = self.create()
        state = self.event(self.event(state))
        state = self.event(state, "text", text="앞서 말씀드린 위치를 정정합니다.")
        self.assertTrue(state["manual"])
        self.assertTrue(all(d["status"] == "WAIT" for d in state["decisions"].values()))

    def test_reroute_and_dispatch_application(self):
        state = self.event(self.create("reroute"))
        self.assertEqual(state["decisions"]["vehicle"]["choice"], "A")
        state = self.event(state)
        self.assertEqual(state["decisions"]["vehicle"]["choice"], "B")
        state = self.event(state, "dispatch")
        self.assertEqual(state["applied_vehicle"], "B")
        self.assertEqual(next(v["availability"] for v in state["vehicles"] if v["id"] == "B"),
                         "unavailable")
        dup = self.client.post("/api/sessions/"+state["id"]+"/events",
            json={"action":"dispatch","expected_version":state["version"]})
        self.assertEqual(dup.status_code, 409)
        self.assertEqual(self.event(state)["applied_vehicle"], "B")

    def test_hospital_retry_preserves_vehicle_and_reveals_source(self):
        state = self.event(self.event(self.create("outage")))
        self.assertEqual(state["external_status"], "timeout")
        vehicle = state["decisions"]["vehicle"]
        self.assertEqual(vehicle["status"], "COMMIT")
        self.assertEqual(state["decisions"]["hospital"]["status"], "WAIT")
        state = self.event(state, "retry")
        self.assertEqual(state["external_status"], "ok")
        self.assertEqual(state["decisions"]["vehicle"], vehicle)
        self.assertEqual(state["decisions"]["hospital"]["choice"], "H2")
        self.assertTrue(all(item["source_at"] for item in state["hospitals"]))
        following = self.event(state)
        self.assertEqual(following["external_status"], "ok")
        self.assertEqual(following["decisions"]["hospital"]["choice"], "H2")

    def test_correction_during_inference_discards_the_old_reply(self):
        from backend.api import create_app
        from backend.inference import Inference
        from fastapi.testclient import TestClient
        started, finish = Event(), Event()
        class SlowIntake(Inference):
            def decide(self, mode, role, state, criteria, simulation_choices):
                started.set()
                finish.wait(3)
                return super().decide(mode, role, state, criteria, simulation_choices)
        with TestClient(create_app(inference=SlowIntake())) as client:
            state = client.post("/api/sessions", json={"scenario_id":"reroute","mode":"simulation"}).json()
            url = "/api/sessions/"+state["id"]
            try:
                first = client.post(url+"/events", json={"action":"next","expected_version":0})
                self.assertEqual(first.status_code, 202)
                self.assertTrue(started.wait(2))
                second = client.post(url+"/events", json={"action":"text","expected_version":1,
                                                          "text":"이전 신고 위치를 정정합니다."})
                self.assertEqual(second.status_code, 202)
                self.assertEqual(second.json()["version"], 2)
            finally:
                finish.set()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                view = client.get(url).json()
                if any(i["status"] == "STALE" for i in view["history"]):
                    break
                time.sleep(.01)
            self.assertTrue(any(i["status"] == "STALE" for i in view["history"]))
            self.assertTrue(all(d["status"] == "WAIT" for d in view["decisions"].values()))
