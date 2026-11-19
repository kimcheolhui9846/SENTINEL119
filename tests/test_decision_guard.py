"""Black-box tests of the published choice envelope and reviewed simulation inputs."""
import itertools
import math
import unittest

from decision_guard import Guard


LABELS = {
    "dispatch": ("구급", "화재", "구조", "기타"),
    "urgency": ("상", "중", "하"),
    "vehicle": ("V1", "V2"), "hospital": ("H1",),
}


def advance(guard, turn=1, dispatch=("구급",), urgency=("상",), location=("북문", 1),
            vehicles=None, hospitals=None):
    return guard.advance(
        turn, supports={"dispatch": {label: (1,) for label in dispatch},
                        "urgency": {label: (1,) for label in urgency}},
        location=location,
        vehicles={"V1": "available", "V2": "available"} if vehicles is None else vehicles,
        hospitals={"H1": "available"} if hospitals is None else hospitals,
    )


def answer(task, choice):
    labels = LABELS[task]
    others = len(labels) - 1
    distribution = {label: round(0.4 / others, 4) if others else 1.0 for label in labels}
    distribution[choice] = 0.6 if others else 1.0
    return {"answers": {task: {"type": "choice", "choice": choice,
                               "probabilities": distribution, "confidence": 0.5}}}


def apply(guard, task, choice):
    return guard.apply(guard.begin(task), answer(task, choice))


class DecisionGuardTests(unittest.TestCase):
    def setUp(self):
        self.guard = Guard("training-demo")
        advance(self.guard)

    def commit_all(self):
        for task, choice in (("dispatch", "구급"), ("urgency", "상"),
                             ("vehicle", "V1"), ("hospital", "H1")):
            result = apply(self.guard, task, choice)
            self.assertEqual((result.status, result.choice), ("COMMIT", choice))

    def test_positive_controls_and_retained_history(self):
        self.commit_all()
        prior = self.guard.active("vehicle")
        history = self.guard.history
        self.assertEqual(self.guard.active("dispatch").evidence_turn_ids, (1,))
        self.assertIsInstance(history, tuple)
        advance(self.guard, dispatch=())
        for task in ("dispatch", "vehicle", "hospital"):
            self.assertEqual(self.guard.active(task).status, "WAIT")
        self.assertEqual(self.guard.active("urgency").status, "COMMIT")
        self.assertIn(prior, self.guard.history)
        self.assertEqual(self.guard.history[:len(history)], history)

    def test_location_correction_revokes_only_dependent_choices(self):
        self.commit_all()
        advance(self.guard, turn=2, location=("남문", 2))
        self.assertEqual(self.guard.active("vehicle").status, "WAIT")
        self.assertEqual(self.guard.active("hospital").status, "WAIT")
        self.assertEqual(self.guard.active("dispatch").status, "COMMIT")
        self.assertEqual(self.guard.active("urgency").status, "COMMIT")

    def test_unknown_hospital_does_not_stop_the_vehicle(self):
        self.commit_all()
        advance(self.guard, hospitals={"H1": "unknown"})
        self.assertEqual(self.guard.active("hospital").status, "WAIT")
        self.assertEqual(apply(self.guard, "hospital", "H1").status, "WAIT")
        self.assertEqual(self.guard.active("vehicle").status, "COMMIT")
        advance(self.guard, location=None)
        self.assertEqual(apply(self.guard, "vehicle", "V1").status, "WAIT")

    def test_nonmedical_dispatch_is_out_of_scope(self):
        advance(self.guard, dispatch=("화재",))
        self.assertEqual(apply(self.guard, "dispatch", "화재").status, "COMMIT")
        for task in ("vehicle", "hospital"):
            self.assertEqual(self.guard.active(task).status, "OUT_OF_SCOPE")

    def test_wrong_guard_duplicate_and_malformed_late_reply(self):
        other = Guard("training-demo")
        advance(other)
        valid = answer("urgency", "상")
        old, current = self.guard.begin("urgency"), self.guard.begin("urgency")
        for ticket in (other.begin("urgency"), old):
            self.assertEqual(self.guard.apply(ticket, {}).status, "STALE")
        self.assertEqual(self.guard.apply(current, valid).status, "COMMIT")
        self.assertEqual(self.guard.apply(current, {}).status, "STALE")
        self.assertEqual(self.guard.active("urgency").choice, "상")

    def test_independent_requests_complete_in_reverse_order(self):
        dispatch, urgency = self.guard.begin("dispatch"), self.guard.begin("urgency")
        self.assertEqual(self.guard.apply(urgency, answer("urgency", "상")).status, "COMMIT")
        self.assertEqual(self.guard.apply(dispatch, answer("dispatch", "구급")).status, "COMMIT")

    def test_caller_mutation_does_not_rewrite_reviewed_facts(self):
        supports = {"dispatch": {"구급": (1,)}}
        vehicles = {"V1": "available", "V2": "unavailable"}
        self.guard.advance(1, supports=supports, location=("북문", 1),
                           vehicles=vehicles, hospitals={})
        supports["dispatch"].clear()
        vehicles["V1"] = "unavailable"
        self.assertEqual(apply(self.guard, "dispatch", "구급").status, "COMMIT")
        self.assertEqual(apply(self.guard, "vehicle", "V1").status, "COMMIT")

    def test_invalid_advance_is_atomic_even_for_pending_ticket(self):
        self.commit_all()
        for bad in ({"turn": 5}, {"turn": True}, {"dispatch": ("unknown",)},
                    {"location": ("", 1)}, {"location": ("북문", 2)},
                    {"vehicles": {"V1": "maybe"}}, {"vehicles": {True: "available"}},
                    {"hospitals": {str(i): "available" for i in range(256)}}):
            with self.subTest(snapshot=next(iter(bad))):
                ticket = self.guard.begin("urgency")
                with self.assertRaises(ValueError):
                    advance(self.guard, **bad)
                self.assertEqual(self.guard.apply(ticket, answer("urgency", "상")).status, "COMMIT")
                self.assertEqual(self.guard.active("vehicle").choice, "V1")

    def test_malformed_latest_reply_cannot_launder_a_previous_commit(self):
        for change in ({"type": "score"}, {"choice": "중"}, {"choice": "X"},
                       {"probabilities": {"상": .6, "중": .4}},
                       {"probabilities": {"상": .6, "중": True, "하": 0}},
                       {"probabilities": {"상": math.nan, "중": .2, "하": .2}},
                       {"probabilities": {"상": .6, "중": .6, "하": .6}}):
            with self.subTest(change=change):
                self.assertEqual(apply(self.guard, "urgency", "상").status, "COMMIT")
                raw = answer("urgency", "상")
                raw["answers"]["urgency"].update(change)
                self.assertEqual(self.guard.apply(self.guard.begin("urgency"), raw).status, "ERROR")
                self.assertEqual(self.guard.active("urgency").status, "WAIT")

    def test_rounded_ties_are_valid(self):
        raw = answer("urgency", "상")
        raw["answers"]["urgency"]["probabilities"] = dict.fromkeys(LABELS["urgency"], .3333)
        self.assertEqual(self.guard.apply(self.guard.begin("urgency"), raw).status, "COMMIT")

    def test_all_24_urgency_support_and_choice_combinations(self):
        for n in range(4):
            for supported in itertools.combinations(LABELS["urgency"], n):
                for choice in LABELS["urgency"]:
                    with self.subTest(supported=supported, choice=choice):
                        advance(self.guard, urgency=supported)
                        expected = "COMMIT" if supported == (choice,) else "WAIT"
                        self.assertEqual(apply(self.guard, "urgency", choice).status, expected)
