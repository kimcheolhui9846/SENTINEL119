"""Current waiting reasons in the session view."""

from time import monotonic, sleep
import unittest
from unittest.mock import patch

from backend.inference import Inference, InferenceError
from backend.scenarios import get_frame
from backend.sessions import SessionService


class SessionReasons(unittest.TestCase):
    def service(self, inference=None):
        service = SessionService(inference)
        self.addCleanup(service.close)
        return service

    def event(self, service, session, action="next", text=None, **fields):
        state = service.event(session["id"], action, session["version"], text, **fields)
        deadline = monotonic() + 3
        while state["processing"] and monotonic() < deadline:
            sleep(.01)
            state = service.get(session["id"])
        self.assertFalse(state["processing"])
        return state

    def assert_wait_reason(self, state, task, expected):
        decision = state["decisions"][task]
        self.assertEqual(decision["status"], "WAIT")
        self.assertEqual(decision["reason"], expected)
        self.assertEqual(decision["missing_information"], [expected])

    def test_staged_hospital_names_missing_information_while_vehicle_commits(self):
        service = self.service()
        session = service.create("staged", "simulation")
        state = self.event(service, self.event(service, session))
        self.assertEqual(state["decisions"]["vehicle"]["status"], "COMMIT")
        self.assert_wait_reason(state, "hospital", "병원 수용 정보가 아직 확인되지 않았습니다.")

    def test_confirmation_waiting_for_location_and_no_available_candidates(self):
        def unavailable_frame(scenario_id, step, hospital_retry=False):
            result = get_frame(scenario_id, step, hospital_retry)
            if step == 1:
                result["location"] = None
            if step == 2:
                for candidate in result["hospitals"]:
                    candidate["availability"] = "unavailable"
            return result

        with patch("backend.sessions.get_frame", unavailable_frame):
            service = self.service()
            first = self.event(service, service.create("reroute", "simulation"))
            for task in ("vehicle", "hospital"):
                self.assert_wait_reason(first, task, "신고 위치 확인을 기다립니다.")
            second = self.event(service, first)
            self.assertEqual(second["decisions"]["vehicle"]["status"], "COMMIT")
            self.assert_wait_reason(second, "hospital", "현재 가용한 병원 후보가 없습니다.")

    def test_timeout_and_manual_text_have_distinct_waiting_reasons(self):
        service = self.service()
        session = service.create("outage", "simulation")
        outage = self.event(service, self.event(service, session))
        self.assertEqual(outage["decisions"]["vehicle"]["status"], "COMMIT")
        self.assert_wait_reason(outage, "hospital",
                                "병원 정보를 조회하지 못했습니다. 병원 상태를 다시 확인할 수 있습니다.")
        manual = self.event(service, outage, "text", "앞서 말씀드린 위치를 정정합니다.")
        for task in manual["decisions"]:
            self.assert_wait_reason(manual, task, "직접 입력의 근거가 검토될 때까지 기다립니다.")

    def test_failed_hospital_model_has_error_reason_and_preserves_vehicle(self):
        class FailedHospital(Inference):
            def decide(self, mode, role, state, criteria, simulation_choices):
                if role == "hospital":
                    raise InferenceError("model_timeout")
                return super().decide(mode, role, state, criteria, simulation_choices)

        service = self.service(FailedHospital())
        state = self.event(service, service.create("reroute", "simulation"))
        self.assertEqual(state["decisions"]["vehicle"]["status"], "COMMIT")
        self.assertEqual(state["decisions"]["hospital"]["execution"], "error")
        self.assert_wait_reason(state, "hospital", "모델 응답 대기 시간을 초과했습니다.")

    def test_applied_dispatch_waiting_reason_describes_the_completed_action(self):
        service = self.service()
        ready = self.event(service, service.create("reroute", "simulation"))
        recommendation = ready["decisions"]["vehicle"]
        applied = self.event(service, ready, "dispatch", expected_vehicle_id=recommendation["choice"],
                             expected_vehicle_revision=recommendation["recommendation_revision"])
        self.assertEqual(applied["applied_vehicle"], "A")
        decision = applied["decisions"]["vehicle"]
        self.assertEqual(decision["status"], "WAIT")
        self.assertEqual(decision["reason"], "모의 출동이 적용되었습니다.")
        self.assertEqual(decision["missing_information"], [])
