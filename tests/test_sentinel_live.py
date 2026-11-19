"""Confidence commits, hierarchy and the SENTINEL_ZORO adapter used by live mode."""
import os
import time
import unittest
from unittest import mock

from decision_guard import Guard
from backend import inference
from backend.sessions import SessionService


def envelope(task, probabilities):
    return {"answers": {task: {"type": "choice", "choice": max(probabilities, key=probabilities.get),
                               "probabilities": probabilities}}}


DISPATCH = {"구급": 0.95, "화재": 0.02, "구조": 0.02, "기타": 0.01}
UNSURE = {"구급": 0.5, "화재": 0.3, "구조": 0.1, "기타": 0.1}
FIRE = {"구급": 0.02, "화재": 0.95, "구조": 0.02, "기타": 0.01}
HIGH = {"상": 0.9, "중": 0.07, "하": 0.03}


def advance(guard, turn):
    guard.advance(turn, supports={}, location=None, vehicles={}, hospitals={})


class ThresholdGuardTests(unittest.TestCase):
    def setUp(self):
        self.guard = Guard("s", thresholds={"dispatch": 0.9, "urgency": 0.85})

    def decide(self, task, probabilities):
        return self.guard.apply(self.guard.begin(task), envelope(task, probabilities))

    def test_commits_at_threshold_and_stays_through_weaker_turns(self):
        advance(self.guard, 1)
        self.assertEqual(self.decide("dispatch", UNSURE).status, "WAIT")
        advance(self.guard, 2)
        result = self.decide("dispatch", DISPATCH)
        self.assertEqual((result.status, result.choice, result.evidence_turn_ids), ("COMMIT", "구급", (2,)))
        advance(self.guard, 3)
        self.assertEqual(self.guard.active("dispatch").status, "COMMIT")
        result = self.decide("dispatch", UNSURE)
        self.assertEqual((result.status, result.choice, result.evidence_turn_ids), ("COMMIT", "구급", (2,)))
        advance(self.guard, 4)
        self.assertEqual(self.decide("dispatch", FIRE).choice, "화재")

    def test_urgency_waits_for_dispatch(self):
        advance(self.guard, 1)
        self.assertEqual(self.decide("urgency", HIGH).status, "WAIT")
        self.decide("dispatch", DISPATCH)
        self.assertEqual(self.decide("urgency", HIGH).status, "COMMIT")

    def test_failed_call_keeps_commit(self):
        advance(self.guard, 1)
        self.decide("dispatch", DISPATCH)
        self.guard.apply(self.guard.begin("dispatch"), None)
        self.assertEqual(self.guard.active("dispatch").status, "COMMIT")

    def test_rejects_unknown_threshold_task(self):
        with self.assertRaises(ValueError):
            Guard("s", thresholds={"vehicle": 0.9})


class FakeInference:
    def decide(self, mode, role, state, criteria, choices):
        if role != "intake":
            first = next(iter(criteria[role]))
            probabilities = {key: float(key == first) for key in criteria[role]}
            return envelope(role, probabilities), "base"
        text = state["turns"][-1]["text"]
        dispatch = DISPATCH if "쓰러" in text else UNSURE
        answers = {"dispatch": envelope("dispatch", dispatch)["answers"]["dispatch"]}
        if "urgency" in criteria:
            answers["urgency"] = envelope("urgency", HIGH)["answers"]["urgency"]
        hierarchy = {"ems_subtype": {"labels": {"cardiac_arrest": "심정지", "illness": "질병(중증 외)"},
                                     "probabilities": {"cardiac_arrest": 0.9, "illness": 0.1}},
                     "ems_symptom": {"labels": {"arrest": "심정지·호흡정지", "pain": "기타 통증"},
                                     "probabilities": {"arrest": 0.6, "pain": 0.4}}}
        return {"answers": answers, "hierarchy": hierarchy}, "SENTINEL_ZORO"


class LiveSessionTests(unittest.TestCase):
    def wait(self, service, view):
        for _ in range(200):
            if not view["processing"]:
                return view
            time.sleep(0.02)
            view = service.get(view["id"])
        self.fail("processing did not finish")

    def test_typed_turns_are_judged_with_hierarchy(self):
        service = SessionService(inference=FakeInference())
        self.addCleanup(service.close)
        view = service.create("staged", "live")
        view = self.wait(service, service.event(view["id"], "text", view["version"], text="여보세요"))
        self.assertEqual(view["decisions"]["dispatch"]["status"], "WAIT")
        view = self.wait(service, service.event(view["id"], "text", view["version"], text="어머니가 쓰러졌어요"))
        self.assertEqual((view["decisions"]["dispatch"]["status"], view["decisions"]["dispatch"]["choice"]), ("COMMIT", "구급"))
        self.assertEqual(view["decisions"]["urgency"]["status"], "COMMIT")
        self.assertEqual((view["hierarchy"]["subtype"]["status"], view["hierarchy"]["subtype"]["choice"]), ("COMMIT", "심정지"))
        self.assertEqual(view["hierarchy"]["symptom"]["status"], "WAIT")

    def test_simulation_typed_turns_stay_unjudged(self):
        service = SessionService(inference=FakeInference())
        self.addCleanup(service.close)
        view = service.create("staged", "simulation")
        view = service.event(view["id"], "text", view["version"], text="어머니가 쓰러졌어요")
        self.assertFalse(view["processing"])
        self.assertEqual(view["hierarchy"], {})


class AdapterTests(unittest.TestCase):
    def test_maps_trained_keys_to_console_labels(self):
        stream = {"labels": {"ems_subtype": {"cardiac_arrest": "심정지"}},
                  "results": {"sft": {"model": {"checkpoint": {"path": "zoro-hier-h1b"}},
                                      "probabilities": {"dispatch": {"fire": 0.1, "rescue": 0.1, "ems": 0.7, "other": 0.1},
                                                        "urgency": {"high": 0.2, "medium": 0.5, "low": 0.3},
                                                        "ems_subtype": {"cardiac_arrest": 1.0}}}}}

        async def fake_request(client, method, url, **kwargs):
            self.assertTrue(url.endswith("/v1/stream"))
            self.assertEqual(kwargs["json"]["turns"], [{"speaker": 0, "text": "119입니다"}, {"speaker": 1, "text": "불이야"}])
            return stream

        with mock.patch.dict(os.environ, {"SENTINEL_ZORO_URL": "http://127.0.0.1:9"}), \
                mock.patch.object(inference, "_json_request", fake_request):
            state = {"turns": [{"id": 1, "speaker": "수보요원", "text": "119입니다"},
                               {"id": 2, "speaker": "신고자", "text": "불이야"}]}
            criteria = {"dispatch": {k: k for k in ("화재", "구조", "구급", "기타")},
                        "urgency": {k: k for k in ("상", "중", "하")}}
            result, model_id = inference.Inference().decide("live", "intake", state, criteria, {})
        self.assertEqual(result["answers"]["dispatch"]["choice"], "구급")
        self.assertEqual(result["answers"]["urgency"]["choice"], "중")
        self.assertEqual(set(result["answers"]["dispatch"]["probabilities"]), {"화재", "구조", "구급", "기타"})
        self.assertIn("ems_subtype", result["hierarchy"])
        self.assertEqual(model_id, "SENTINEL_ZORO · zoro-hier-h1b")


if __name__ == "__main__":
    unittest.main()
