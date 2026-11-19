"""Public contract for the local dispatch demo. Fixtures never need real model weights."""
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Barrier, Event
from importlib.util import find_spec
from unittest.mock import patch


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
        if action == "dispatch":
            fields = {**self.dispatch_proof(state), **fields}
        url = "/api/sessions/" + state["id"]
        response = self.client.post(url + "/events", json={
            "action":action, "expected_version":state["version"], **fields})
        self.assertEqual(response.status_code, 202, response.text)
        result = response.json()
        return self.wait_for_completion(self.client, url) if result["processing"] else result

    @staticmethod
    def dispatch_proof(state):
        vehicle = state["decisions"]["vehicle"]
        return {"expected_vehicle_id": vehicle["choice"],
                "expected_vehicle_revision": vehicle["recommendation_revision"]}

    def test_dispatch_requires_current_vehicle_id_and_revision(self):
        ready = self.event(self.create("reroute"))
        url = "/api/sessions/" + ready["id"]
        proof = self.dispatch_proof(ready)
        for fields in ({}, {"expected_vehicle_id": "A"},
                       {"expected_vehicle_revision": proof["expected_vehicle_revision"]},
                       {**proof, "expected_vehicle_id": "B"},
                       {**proof, "expected_vehicle_revision": proof["expected_vehicle_revision"] + 1}):
            with self.subTest(fields=fields):
                response = self.client.post(url + "/events", json={
                    "action": "dispatch", "expected_version": ready["version"], **fields,
                })
                self.assertEqual(response.status_code, 409, response.text)
                self.assertEqual(response.json(), {"detail": "stale_vehicle_recommendation"})
                self.assertIsNone(self.client.get(url).json()["applied_vehicle"])
        self.assertEqual(self.event(ready, "dispatch")["applied_vehicle"], "A")

    def test_dispatch_proof_does_not_allow_an_expired_vehicle(self):
        from backend.scenarios import get_frame

        def expiring_vehicle(scenario_id, step, hospital_retry=False):
            frame = get_frame(scenario_id, step, hospital_retry)
            frame["vehicles"][0]["expires_at"] = "2026-11-19T05:01:00+00:00"
            return frame

        with patch("backend.sessions.get_frame", expiring_vehicle):
            with patch("backend.sessions._now", return_value=datetime(2026, 11, 19, 5, 0, tzinfo=timezone.utc)):
                ready = self.event(self.create("reroute"))
            self.assertEqual(ready["decisions"]["vehicle"]["choice"], "A")
            url = "/api/sessions/" + ready["id"]
            with patch("backend.sessions._now", return_value=datetime(2026, 11, 19, 5, 1, tzinfo=timezone.utc)):
                response = self.client.post(url + "/events", json={
                    "action": "dispatch", "expected_version": ready["version"], **self.dispatch_proof(ready),
                })
            self.assertEqual(response.status_code, 409, response.text)
            self.assertEqual(response.json(), {"detail": "vehicle_not_ready"})
            self.assertIsNone(self.client.get(url).json()["applied_vehicle"])

    def test_only_one_concurrent_dispatch_can_apply_a_viewed_recommendation(self):
        ready = self.event(self.create("reroute"))
        url = "/api/sessions/" + ready["id"]
        simultaneous = Barrier(2, timeout=2)

        def apply():
            simultaneous.wait()
            return self.client.post(url + "/events", json={
                "action": "dispatch", "expected_version": ready["version"], **self.dispatch_proof(ready),
            })

        with ThreadPoolExecutor(max_workers=2) as executor:
            requests = [executor.submit(apply) for _ in range(2)]
            responses = sorted((future.result() for future in requests), key=lambda result: result.status_code)
        accepted, rejected = responses
        self.assertEqual(accepted.status_code, 202, accepted.text)
        self.assertEqual(rejected.status_code, 409, rejected.text)
        self.assertEqual(rejected.json(), {"detail": "stale_version"})
        state = self.client.get(url).json()
        self.assertEqual(state["applied_vehicle"], "A")
        self.assertEqual(sum(row["status"] == "APPLIED" for row in state["history"]), 1)

    def test_inflight_vehicle_recommendation_invalidates_earlier_dispatch_proof(self):
        from backend.api import create_app
        from backend.inference import Inference
        from fastapi.testclient import TestClient

        for refreshed_choice in ("A", "B"):
            with self.subTest(refreshed_choice=refreshed_choice):
                started, finish = Event(), Event()

                class DelayedVehicle(Inference):
                    vehicle_calls = 0

                    def decide(self, mode, role, state, criteria, simulation_choices):
                        if role == "vehicle":
                            self.vehicle_calls += 1
                            if self.vehicle_calls == 3:
                                started.set()
                                if not finish.wait(3):
                                    raise RuntimeError("Test did not release inference")
                                simulation_choices["vehicle"] = refreshed_choice
                        return super().decide(mode, role, state, criteria, simulation_choices)

                with TestClient(create_app(inference=DelayedVehicle())) as client:
                    session = client.post("/api/sessions", json={"scenario_id": "reroute", "mode": "simulation"}).json()
                    url = "/api/sessions/" + session["id"]
                    completed = []
                    for version in (0, 1):
                        response = client.post(url + "/events", json={"action": "next", "expected_version": version})
                        self.assertEqual(response.status_code, 202)
                        completed.append(self.wait_for_completion(client, url))
                    try:
                        response = client.post(url + "/events", json={"action": "next", "expected_version": 2})
                        self.assertEqual(response.status_code, 202)
                        self.assertTrue(started.wait(2))
                        shown = client.get(url).json()
                        self.assertEqual(shown["decisions"]["vehicle"]["choice"], "B")
                    finally:
                        finish.set()
                    refreshed = self.wait_for_completion(client, url)
                    self.assertEqual(shown["version"], refreshed["version"])
                    self.assertEqual(refreshed["decisions"]["vehicle"]["choice"], refreshed_choice)
                    stale = client.post(url + "/events", json={
                        "action": "dispatch", "expected_version": shown["version"], **self.dispatch_proof(shown),
                    })
                    self.assertEqual(stale.status_code, 409, stale.text)
                    self.assertEqual(stale.json(), {"detail": "stale_vehicle_recommendation"})
                    self.assertGreater(refreshed["decisions"]["vehicle"]["recommendation_revision"],
                                       shown["decisions"]["vehicle"]["recommendation_revision"])
                    if refreshed_choice == "A":
                        self.assertEqual(completed[0]["decisions"]["vehicle"]["choice"], "A")
                        earlier_a = client.post(url + "/events", json={
                            "action": "dispatch", "expected_version": refreshed["version"],
                            **self.dispatch_proof(completed[0]),
                        })
                        self.assertEqual(earlier_a.status_code, 409, earlier_a.text)
                        self.assertEqual(earlier_a.json(), {"detail": "stale_vehicle_recommendation"})
                    updated = client.post(url + "/events", json={
                        "action": "dispatch", "expected_version": refreshed["version"], **self.dispatch_proof(refreshed),
                    })
                    self.assertEqual(updated.status_code, 202, updated.text)
                    self.assertEqual(updated.json()["applied_vehicle"], refreshed_choice)

    def wait_for_completion(self, client, url):
        deadline = time.monotonic() + 3
        while True:
            result = client.get(url).json()
            if not result["processing"]:
                return result
            self.assertLess(time.monotonic(), deadline)
            time.sleep(.01)

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
        self.assertEqual(dup.json(), {"detail": "dispatch_already_applied"})
        self.assertEqual(self.event(state)["applied_vehicle"], "B")

    def test_hospital_retry_preserves_vehicle_and_reveals_source(self):
        before = self.event(self.create("outage"))
        state = self.event(before)
        self.assertEqual(state["external_status"], "timeout")
        vehicle = state["decisions"]["vehicle"]
        self.assertEqual(vehicle["status"], "COMMIT")
        self.assertEqual(state["decisions"]["hospital"]["status"], "WAIT")
        observations = {row["id"]: row for row in before["hospitals"]}
        for row in state["hospitals"]:
            self.assertEqual(row["availability"], "unknown")
            self.assertIsNone(row["source_at"])
            if row["id"] == "H1":
                self.assertIsNone(row.get("last_observation"))
            else:
                previous = observations[row["id"]]
                self.assertEqual(row["last_observation"], {
                    "availability": previous["availability"], "source_at": previous["source_at"],
                })
        state = self.event(state, "retry")
        self.assertEqual(state["external_status"], "ok")
        self.assertEqual(state["decisions"]["vehicle"], vehicle)
        self.assertEqual(state["decisions"]["hospital"]["choice"], "H2")
        current = {item["id"]: item for item in state["hospitals"]}
        self.assertEqual(current["H3"]["availability"], "unavailable")
        self.assertIsNone(current["H1"]["source_at"])
        for hospital_id in ("H2", "H3"):
            self.assertGreater(current[hospital_id]["source_at"], observations[hospital_id]["source_at"])
            self.assertIsNone(current[hospital_id].get("last_observation"))
        following = self.event(state)
        self.assertEqual(following["external_status"], "ok")
        self.assertEqual(following["decisions"]["hospital"]["choice"], "H2")
        for hospital_id in ("H2", "H3"):
            next_row = next(item for item in following["hospitals"] if item["id"] == hospital_id)
            self.assertGreater(next_row["source_at"], current[hospital_id]["source_at"])

    def test_operations_only_turns_do_not_confirm_urgency(self):
        for scenario, vehicle_choice in (("reroute", "B"), ("outage", "A")):
            with self.subTest(scenario=scenario):
                state = self.event(self.event(self.create(scenario)))
                self.assertEqual(state["decisions"]["urgency"]["status"], "WAIT")
                self.assertIsNone(state["decisions"]["urgency"]["choice"])
                self.assertEqual(state["decisions"]["urgency"]["evidence_turn_ids"], [])
                self.assertEqual(state["decisions"]["vehicle"]["choice"], vehicle_choice)

    def test_staged_urgency_cites_clinical_assessment_and_survives_location_correction(self):
        state = self.create("staged")
        for _ in range(4):
            state = self.event(state)
        self.assertIn("의식", state["turns"][2]["text"])
        urgency = state["decisions"]["urgency"]
        self.assertEqual(urgency["choice"], "중")
        self.assertEqual(urgency["evidence_turn_ids"], [3])

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
