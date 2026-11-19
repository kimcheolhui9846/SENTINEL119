import json
import unittest
from copy import deepcopy

from shadow_harness import decide_shadow


EMS = "성수동에서 사람이 갑자기 쓰러졌어요. 구급차를 보내 주세요."
EMS_POLITE = "성수동에서 사람이 갑자기 쓰러졌어요. 구급차를 보내 주실래요?"
FIRE = "성수동 건물에 불이 났어요. 연기와 불꽃이 보여요. 소방차를 보내 주세요."
CANCEL = "정정할게요. 연습 상황이었고 구급차 요청을 취소합니다."
FIRE_UPDATE = "불길이 번지고 있어요. 건물 밖으로 대피했어요."
EMS_SCORES = {"ems": .99, "fire": .01, "rescue": 0, "other": 0}
FIRE_SCORES = {"ems": .01, "fire": .99, "rescue": 0, "other": 0}
UNITS = {"ems": ["A2", "A1"], "fire": ["F1"]}
MODEL = {"model_id": "mock-only", "adapter_version": "demo-v1"}


def caller(text):
    return {"speaker": "신고자", "text": text}


def operator(text):
    return {"speaker": "수보요원", "text": text}


class ShadowHarnessTest(unittest.TestCase):
    def test_record_actual_evidence_location_independent_of_fixture(self):
        for request in (EMS, EMS_POLITE):
            for preceding_turns in range(3):
                with self.subTest(request=request, preceding_turns=preceding_turns):
                    turns = [operator("말씀해 주세요.")] * preceding_turns + [caller(request)]
                    result = decide_shadow(turns, EMS_SCORES, UNITS, **MODEL)
                    self.assertEqual(result["evidence_turn"], preceding_turns + 1)
                    self.assertEqual(result["evidence_span"], request)
                    self.assertEqual(result["category"], "ems")
                    self.assertEqual(result["proposed_unit"], "A2")

    def test_empty_fleet_leaves_category_confirmed(self):
        result = decide_shadow([caller(EMS)], EMS_SCORES, {"ems": []}, **MODEL)
        self.assertEqual(result["category"], "ems")
        self.assertIsNone(result["proposed_unit"])
        self.assertEqual(result["reason"], "no_eligible_unit")

    def test_cancelled_request_rejects_late_high_scores_and_latches_review(self):
        for turns in ([caller(EMS), caller(CANCEL)], [caller(EMS), caller(CANCEL), caller(EMS)]):
            with self.subTest(turns=turns):
                result = decide_shadow(turns, EMS_SCORES, UNITS, **MODEL)
                self.assertIsNone(result["category"])
                self.assertIsNone(result["proposed_unit"])
                self.assertEqual(result["reason"], "cancelled_request")
                self.assertTrue(result["needs_review"])

    def test_unreviewed_caller_turn_does_not_refresh_old_evidence(self):
        for ambiguous in (f"{EMS}라는 훈련 문장을 읽었어요.", "구급차를 보낼 상황은 아닙니다.", FIRE):
            with self.subTest(ambiguous=ambiguous):
                result = decide_shadow([caller(EMS), caller(ambiguous), caller(EMS)], EMS_SCORES, UNITS, **MODEL)
                self.assertIsNone(result["proposed_unit"])
                self.assertIsNone(result["category"])
                self.assertTrue(result["needs_review"])

    def test_operator_words_and_unreviewed_paraphrase_cannot_confirm(self):
        for turn in (operator(EMS), caller(f"‘{EMS}’"), caller("근처 쓰러진 분 때문에 구급차가 필요해요.")):
            with self.subTest(turn=turn):
                result = decide_shadow([turn], EMS_SCORES, UNITS, **MODEL)
                self.assertIsNone(result["category"])
                self.assertIsNone(result["proposed_unit"])

    def test_unreviewed_turn_before_request_and_safe_updates_do_not_erase_request(self):
        turns = [caller("여보세요."), caller(FIRE), operator("계속 말씀하세요."), caller(FIRE_UPDATE)]
        result = decide_shadow(turns, FIRE_SCORES, UNITS, **MODEL)
        self.assertEqual(result["category"], "fire")
        self.assertEqual(result["proposed_unit"], "F1")
        self.assertEqual(result["evidence_turn"], 2)

    def test_applied_unit_is_never_reproposed_and_correction_requests_review(self):
        applied = {"category": "ems", "unit_id": "A1"}
        intact = decide_shadow([caller(EMS)], EMS_SCORES, UNITS, applied_unit=applied, **MODEL)
        corrected = decide_shadow([caller(EMS), caller(CANCEL)], EMS_SCORES, UNITS, applied_unit=applied, **MODEL)
        self.assertEqual(intact["category"], "ems")
        self.assertIsNone(intact["proposed_unit"])
        self.assertEqual(intact["reason"], "already_applied")
        self.assertIsNone(corrected["proposed_unit"])
        self.assertTrue(corrected["needs_review"])
        self.assertEqual(applied, {"category": "ems", "unit_id": "A1"})

        different_category = decide_shadow([caller(FIRE)], FIRE_SCORES, UNITS, applied_unit=applied, **MODEL)
        self.assertEqual(different_category["reason"], "applied_category_conflict")
        self.assertTrue(different_category["needs_review"])
        self.assertIsNone(different_category["proposed_unit"])

    def test_invalid_mandatory_distribution_fails_closed(self):
        for scores in ({}, {"ems": 1.0}, {**EMS_SCORES, "extra": 0}, {**EMS_SCORES, "ems": float("nan")},
                       {**EMS_SCORES, "ems": float("inf")}, {**EMS_SCORES, "ems": True, "fire": 0},
                       {**EMS_SCORES, "ems": .8}, {**EMS_SCORES, "other": -.1}):
            with self.subTest(scores=scores):
                result = decide_shadow([caller(EMS)], scores, UNITS, **MODEL)
                self.assertEqual(result["reason"], "invalid_dispatch_scores")
                self.assertIsNone(result["proposed_unit"])
                self.assertIsNone(result["category"])

    def test_score_threshold_ties_and_model_disagreement_wait(self):
        for scores in (FIRE_SCORES, {**EMS_SCORES, "ems": .5, "fire": .5},
                       {**EMS_SCORES, "ems": .84, "fire": .16}, {**EMS_SCORES, "ems": .01, "fire": 0, "other": .99}):
            with self.subTest(scores=scores):
                result = decide_shadow([caller(EMS)], scores, UNITS, **MODEL)
                self.assertIsNone(result["category"])
                self.assertIsNone(result["proposed_unit"])
        result = decide_shadow([caller(EMS)], {**EMS_SCORES, "ems": .85, "fire": .15}, UNITS, **MODEL)
        self.assertEqual(result["proposed_unit"], "A2")

    def test_optional_head_wait_does_not_block_coarse_and_result_is_json_serializable(self):
        turns, dispatch_scores, eligible_units = [caller(EMS)], dict(EMS_SCORES), deepcopy(UNITS)
        for optional in (None, {}, {"cardiac_arrest": float("inf")}):
            with self.subTest(optional=optional):
                result = decide_shadow(turns, dispatch_scores, eligible_units, subtype_scores=optional, **MODEL)
                self.assertEqual(result["category"], "ems")
                self.assertEqual(result["subtype_status"], "wait")
                self.assertTrue(result["policy_version"])
                self.assertEqual(result["model_id"], MODEL["model_id"])
                self.assertEqual(result["adapter_version"], MODEL["adapter_version"])
                self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        self.assertEqual(turns, [caller(EMS)])
        self.assertEqual(dispatch_scores, EMS_SCORES)
        self.assertEqual(eligible_units, UNITS)

        complete = dict.fromkeys(("severe_illness", "illness", "cardiac_arrest", "injury", "accident", "pregnancy",
                                  "poisoning", "other_ems"), .125)
        result = decide_shadow(turns, dispatch_scores, eligible_units, subtype_scores=complete, **MODEL)
        self.assertEqual(result["subtype_status"], "available")
        self.assertEqual(result["proposed_unit"], "A2")

    def test_invalid_turn_resource_or_provenance_input_is_rejected(self):
        with self.assertRaises(ValueError):
            decide_shadow([], EMS_SCORES, UNITS, **MODEL)
        with self.assertRaises(ValueError):
            decide_shadow([{"speaker": "system", "text": EMS}], EMS_SCORES, UNITS, **MODEL)
        with self.assertRaises(ValueError):
            decide_shadow([caller(EMS)], EMS_SCORES, {"ems": "A2"}, **MODEL)
        with self.assertRaises(ValueError):
            decide_shadow([caller(EMS)], EMS_SCORES, UNITS, model_id="", adapter_version="demo-v1")


if __name__ == "__main__":
    unittest.main()
