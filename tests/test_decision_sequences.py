"""Counterexamples from synthetic, server-reviewed facts. No model or network needed."""

import itertools
import random
import unittest
from threading import Event, Thread

from decision_guard import Guard


DISPATCH = ("화재", "구조", "구급", "기타")
URGENCY = ("상", "중", "하")


def reply(task, choice, labels):
    return {"answers": {task: {
        "type": "choice", "choice": choice, "confidence": 1.0,
        "probabilities": {label: float(choice == label) for label in labels},
    }}}


def advance(guard, turn, *, support="구급", location=("북문", 1),
            vehicle="available", hospital="available"):
    guard.advance(
        turn, supports={"dispatch": {support: (turn,)} if support else {},
                        "urgency": {"상": (1,)}},
        location=location, vehicles={"V": vehicle}, hospitals={"H": hospital},
    )


class DecisionSequences(unittest.TestCase):
    def test_expired_reply_cannot_commit_even_without_a_new_turn(self):
        now = [10.0]
        guard = Guard("demo", max_response_age_s=2, clock=lambda: now[0])
        advance(guard, 1)
        self.assertEqual(guard.apply(guard.begin("dispatch"),
                                     reply("dispatch", "구급", DISPATCH)).status, "COMMIT")
        old = guard.begin("dispatch")
        now[0] = 12.0
        self.assertEqual(guard.apply(old, reply("dispatch", "구급", DISPATCH)).status, "ERROR")
        self.assertEqual(guard.active("dispatch").status, "WAIT")
        self.assertEqual(guard.apply(old, {}).status, "STALE")
        fresh = guard.begin("dispatch")
        now[0] = 13.999
        self.assertEqual(guard.apply(fresh, reply("dispatch", "구급", DISPATCH)).status, "COMMIT")

    def test_late_reply_reports_original_snapshot_for_logging(self):
        guard = Guard("demo")
        advance(guard, 1)
        old = guard.begin("hospital")
        advance(guard, 2, hospital="unknown")
        result = guard.apply(old, {})
        self.assertEqual((result.status, result.task, result.turn_id, result.revision),
                         ("STALE", "hospital", 1, 1))
        self.assertEqual(guard.active("hospital").status, "WAIT")

    def test_same_turn_snapshot_refresh_revokes_inflight_reply(self):
        guard = Guard("demo")
        advance(guard, 1)
        old = guard.begin("dispatch")
        advance(guard, 1)
        self.assertEqual(guard.apply(old, reply("dispatch", "구급", DISPATCH)).status, "STALE")

    def test_changing_an_alternative_reopens_only_its_recommendation(self):
        for updated in ({"A": "available"}, {"A": "available", "B": "available"},
                        {"A": "available", "B": "unknown", "C": "available"}):
            with self.subTest(updated=updated):
                guard = Guard("demo")
                def refresh(vehicles):
                    guard.advance(1, supports={"dispatch": {"구급": [1]}}, location=("북문", 1),
                                  vehicles=vehicles, hospitals={"H": "available"})
                refresh({"A": "available", "B": "unknown"})
                guard.apply(guard.begin("dispatch"), reply("dispatch", "구급", DISPATCH))
                guard.apply(guard.begin("hospital"), reply("hospital", "H", ("H",)))
                guard.apply(guard.begin("vehicle"), reply("vehicle", "A", ("A", "B")))
                refresh(updated)
                self.assertEqual(guard.active("vehicle").status, "WAIT")
                self.assertEqual(guard.active("hospital").choice, "H")
                self.assertEqual(guard.active("dispatch").choice, "구급")
                self.assertEqual(guard.apply(guard.begin("vehicle"),
                                            reply("vehicle", "A", tuple(updated))).status, "COMMIT")

    def test_withdrawal_is_visible_in_the_decision_history(self):
        guard = Guard("demo")
        advance(guard, 1)
        guard.apply(guard.begin("dispatch"), reply("dispatch", "구급", DISPATCH))
        guard.apply(guard.begin("vehicle"), reply("vehicle", "V", ("V",)))
        advance(guard, 2, support=None)
        self.assertEqual([d.status for d in guard.history if d.task == "vehicle"][-2:],
                         ["COMMIT", "WAIT"])

    def test_reordered_candidates_preserve_the_active_recommendation(self):
        guard = Guard("demo")
        def refresh(candidate_ids):
            guard.advance(1, supports={"dispatch": {"구급": [1]}}, location=("북문", 1),
                          vehicles=dict.fromkeys(candidate_ids, "available"), hospitals={})
        refresh(("A", "B"))
        guard.apply(guard.begin("dispatch"), reply("dispatch", "구급", DISPATCH))
        guard.apply(guard.begin("vehicle"), reply("vehicle", "B", ("A", "B")))
        refresh(("B", "A"))
        self.assertEqual(guard.active("vehicle").choice, "B")

    def test_old_timeout_cannot_withdraw_the_newer_commit(self):
        now = [0.0]
        guard = Guard("demo", max_response_age_s=2, clock=lambda: now[0])
        advance(guard, 1)
        old = guard.begin("dispatch")
        now[0] = 3.0
        guard.apply(guard.begin("dispatch"), reply("dispatch", "구급", DISPATCH))
        self.assertEqual(guard.apply(old, {}).status, "STALE")
        self.assertEqual(guard.active("dispatch").choice, "구급")

    def test_default_deadline_is_five_seconds(self):
        now = [0.0]
        guard = Guard("demo", clock=lambda: now[0])
        advance(guard, 1)
        ticket = guard.begin("dispatch")
        now[0] = 5.0
        self.assertEqual(guard.apply(ticket, reply("dispatch", "구급", DISPATCH)).status, "ERROR")
        self.assertEqual(guard.active("dispatch").status, "WAIT")

    def test_empty_candidates_and_errors_respect_dispatch_scope(self):
        guard = Guard("demo")
        for label, expected in (("구급", "WAIT"), ("화재", "OUT_OF_SCOPE")):
            guard.advance(1, supports={"dispatch": {label: [1]}}, location=("북문", 1),
                          vehicles={}, hospitals={"H": "available"})
            guard.apply(guard.begin("dispatch"), reply("dispatch", label, DISPATCH))
            self.assertEqual(guard.apply(guard.begin("vehicle"), {}).status, expected)
            self.assertEqual(guard.apply(guard.begin("hospital"), {}).status, "ERROR")
            self.assertEqual(guard.active("hospital").status, expected)

    def test_location_evidence_turn_changed_even_when_text_is_identical(self):
        guard = Guard("demo")
        for turn in (1, 2):
            guard.advance(turn, supports={"dispatch": {"구급": [1]}}, location=("북문", turn),
                          vehicles={"V": "available"}, hospitals={})
            if turn == 1:
                guard.apply(guard.begin("dispatch"), reply("dispatch", "구급", DISPATCH))
                self.assertEqual(guard.apply(guard.begin("vehicle"), reply("vehicle", "V", ("V",))).status,
                                 "COMMIT")
        self.assertEqual(guard.active("vehicle").status, "WAIT")
        self.assertEqual(guard.active("dispatch").choice, "구급")

    def test_invalid_evidence_does_not_consume_a_pending_decision(self):
        for invalid_ids in ([2], [True], [0], [1, 1], []):
            with self.subTest(invalid_ids=invalid_ids):
                guard = Guard("demo")
                advance(guard, 1)
                old = guard.begin("dispatch")
                with self.assertRaises(ValueError):
                    guard.advance(1, supports={"dispatch": {"구급": invalid_ids}},
                                  location=None, vehicles={}, hospitals={})
                self.assertEqual(guard.apply(old, reply("dispatch", "구급", DISPATCH)).status,
                                 "COMMIT")

    def test_dependency_change_between_request_and_delivery(self):
        guard = Guard("demo")
        advance(guard, 1)
        guard.apply(guard.begin("dispatch"), reply("dispatch", "구급", DISPATCH))
        issued = Event()
        deliver = Event()
        outcomes = []

        def worker():
            ticket = guard.begin("vehicle")
            issued.set()
            if deliver.wait(5):
                outcomes.append(guard.apply(ticket, reply("vehicle", "V", ("V",))).status)

        thread = Thread(target=worker, daemon=True)
        thread.start()
        try:
            self.assertTrue(issued.wait(5))
            # Same turn, no change to the source snapshot. Both endpoints say 구급.
            for choice in ("구조", "구급"):
                guard.apply(guard.begin("dispatch"), reply("dispatch", choice, DISPATCH))
        finally:
            deliver.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(outcomes, ["STALE"])
        self.assertEqual(guard.active("vehicle").status, "WAIT")

    def test_late_replies_in_all_delivery_orders(self):
        for order in itertools.permutations(range(3)):
            with self.subTest(order=order):
                guard = Guard("demo")
                advance(guard, 1)
                tickets = [guard.begin("urgency"), guard.begin("urgency")]
                advance(guard, 2)
                tickets.append(guard.begin("urgency"))
                for index in order:
                    result = guard.apply(tickets[index], reply("urgency", "상", URGENCY))
                    self.assertEqual(result.status, "COMMIT" if index == 2 else "STALE")
                self.assertEqual(guard.active("urgency").choice, "상")

    def test_eight_thousand_synthetic_corrections(self):
        # The oracle uses the final facts in each generated story and never
        # calls a production validator or derives expectations from its output.
        statuses = ("available", "unavailable", "unknown")
        for seed in range(8000):
            rng = random.Random(seed)
            guard = Guard("demo")
            advance(guard, 1)
            guard.apply(guard.begin("dispatch"), reply("dispatch", "구급", DISPATCH))
            guard.apply(guard.begin("urgency"), reply("urgency", "상", URGENCY))
            guard.apply(guard.begin("vehicle"), reply("vehicle", "V", ("V",)))
            guard.apply(guard.begin("hospital"), reply("hospital", "H", ("H",)))
            pending = guard.begin(rng.choice(("vehicle", "hospital")))
            support = rng.choice(("구급", "화재", None))
            location = rng.choice((("북문", 1), ("남문", 2), None))
            vehicle, hospital = rng.choices(statuses, k=2)
            advance(guard, 2, support=support, location=location,
                    vehicle=vehicle, hospital=hospital)
            self.assertEqual(guard.active("urgency").choice, "상", seed)
            self.assertEqual(guard.apply(pending, {}).status, "STALE", seed)
            guess = rng.choice(("구급", "화재"))
            guard.apply(guard.begin("dispatch"), reply("dispatch", guess, DISPATCH))
            supported = guess == support
            self.assertEqual(guard.active("dispatch").status,
                             "COMMIT" if supported else "WAIT", seed)
            tasks = [("vehicle", "V", vehicle), ("hospital", "H", hospital)]
            rng.shuffle(tasks)
            for task, candidate, availability in tasks:
                if supported and guess == "화재":
                    expected = "OUT_OF_SCOPE"
                elif supported and location is not None and availability == "available":
                    expected = "COMMIT"
                else:
                    expected = "WAIT"
                guard.apply(guard.begin(task), reply(task, candidate, (candidate,)))
                self.assertEqual(guard.active(task).status, expected, (seed, task))


class CandidateVersions(unittest.TestCase):
    def setUp(self):
        self.guard = Guard("versions")

    def refresh(self, versions, vehicles=None, location=("북문", 1)):
        self.guard.advance(1, supports={"dispatch": {"구급": [1]}}, location=location,
                           vehicles=dict.fromkeys(("A", "B"), "available") if vehicles is None else vehicles,
                           hospitals=dict.fromkeys(("H1", "H2"), "available"),
                           candidate_versions=versions)

    def apply(self, task, choice):
        labels = {"dispatch": DISPATCH, "vehicle": ("A", "B"), "hospital": ("H1", "H2")}
        return self.guard.apply(self.guard.begin(task), reply(task, choice, labels[task]))

    def test_eta_reversal_reopens_only_the_affected_recommendation(self):
        from decision_guard import CandidateVersionTracker
        tracker = CandidateVersionTracker()
        eta_minutes = {"A": 3, "B": 9}
        before = tracker.observe(eta_minutes)
        self.refresh({"vehicle": before, "hospital": 3})
        self.apply("dispatch", "구급")
        self.assertEqual(self.apply("vehicle", min(eta_minutes, key=eta_minutes.get)).choice, "A")
        self.apply("hospital", "H1")
        inflight = self.guard.begin("vehicle")
        eta_minutes = {"A": 12, "B": 4}
        after = tracker.observe(eta_minutes)
        self.assertGreater(after, before)
        self.refresh({"vehicle": after, "hospital": 3})
        self.assertEqual(self.guard.active("vehicle").status, "WAIT")
        self.assertEqual(self.guard.active("hospital").choice, "H1")
        self.assertEqual(self.guard.apply(inflight, reply("vehicle", "A", ("A", "B"))).status, "STALE")
        self.assertEqual(self.apply("vehicle", min(eta_minutes, key=eta_minutes.get)).choice, "B")
        self.refresh({"vehicle": after, "hospital": 3})
        self.assertEqual(self.guard.active("vehicle").choice, "B")
        self.refresh({"vehicle": after, "hospital": 4})
        self.assertEqual(self.guard.active("vehicle").choice, "B")
        self.assertEqual(self.guard.active("hospital").status, "WAIT")

    def test_first_version_supersedes_unversioned_choice(self):
        self.refresh(None)
        self.apply("dispatch", "구급")
        self.apply("vehicle", "A")
        self.apply("hospital", "H1")
        self.refresh({"hospital": 0})
        self.assertEqual(self.guard.active("vehicle").choice, "A")
        self.assertEqual(self.guard.active("hospital").status, "WAIT")
        self.assertEqual(self.apply("hospital", "H2").status, "COMMIT")

    def test_version_omission_rollback_and_recovery_in_all_short_sequences(self):
        for versions in itertools.product((None, 0, 1, 2), repeat=3):
            with self.subTest(versions=versions):
                self.guard = Guard("story")
                self.refresh({"vehicle": 1})
                self.apply("dispatch", "구급")
                self.apply("vehicle", "A")
                self.apply("hospital", "H1")
                high, committed = 1, True
                for version in versions:
                    self.refresh({} if version is None else {"vehicle": version})
                    usable = version is not None and version >= high
                    retained = committed and usable and version == high
                    self.assertEqual(self.guard.active("vehicle").status, "COMMIT" if retained else "WAIT")
                    self.assertEqual(self.guard.active("hospital").choice, "H1")
                    self.assertEqual(self.apply("vehicle", "B").status, "COMMIT" if usable else "WAIT")
                    if usable:
                        high = version
                    committed = usable

    def test_malformed_versions_leave_the_previous_snapshot_and_ticket_intact(self):
        self.refresh({"vehicle": 1})
        self.apply("dispatch", "구급")
        for bad in ({"dispatch": 1}, {"vehicle": True}, {"hospital": -1}, {"vehicle": 1.0}, []):
            with self.subTest(bad=bad):
                ticket = self.guard.begin("vehicle")
                with self.assertRaises(ValueError):
                    self.refresh(bad)
                self.assertEqual(self.guard.apply(ticket, reply("vehicle", "A", ("A", "B"))).status, "COMMIT")

    def test_changed_candidate_map_at_same_version_requires_strictly_newer_version(self):
        self.refresh({"vehicle": 7})
        self.apply("dispatch", "구급")
        self.apply("vehicle", "A")
        self.refresh({"vehicle": 7}, vehicles={"A": "available", "B": "unknown"})
        self.assertEqual(self.guard.active("vehicle").status, "WAIT")
        self.assertEqual(self.apply("vehicle", "A").status, "WAIT")
        self.refresh({"vehicle": 7})
        self.assertEqual(self.apply("vehicle", "A").status, "WAIT")
        self.refresh({"vehicle": 8})
        self.assertEqual(self.guard.active("vehicle").status, "WAIT")
        self.assertEqual(self.apply("vehicle", "A").status, "COMMIT")

    def test_forgotten_version_never_reverts_to_legacy_mode(self):
        self.refresh({"vehicle": 4})
        self.apply("dispatch", "구급")
        for missing in (None, {}, {"hospital": 0}):
            self.refresh({"vehicle": 4})
            self.apply("vehicle", "A")
            old = self.guard.begin("vehicle")
            self.refresh(missing)
            self.assertEqual(self.apply("vehicle", "A").status, "WAIT")
            self.refresh({"vehicle": 4})
            self.assertEqual(self.guard.active("vehicle").status, "WAIT")
            self.assertEqual(self.guard.apply(old, reply("vehicle", "A", ("A", "B"))).status, "STALE")

    def test_invalid_snapshot_does_not_consume_version(self):
        initial = {"vehicle": 4}
        self.refresh(initial)
        initial["vehicle"] = 50
        self.apply("dispatch", "구급")
        ticket = self.guard.begin("vehicle")
        with self.assertRaises(ValueError):
            self.refresh({"vehicle": 8}, location=("북문", 2))
        self.assertEqual(self.guard.apply(ticket, reply("vehicle", "A", ("A", "B"))).status, "COMMIT")
        self.refresh({"vehicle": 5})
        self.assertEqual(self.apply("vehicle", "B").status, "COMMIT")

    def test_tracker_detaches_content_ignores_map_order_and_rejects_nan(self):
        from decision_guard import CandidateVersionTracker
        tracker = CandidateVersionTracker()
        raw = {"A": {"eta": 3, "available": True}, "B": {"eta": 9}}
        first = tracker.observe(raw)
        self.assertEqual(tracker.observe({"B": {"eta": 9}, "A": {"available": True, "eta": 3}}), first)
        for bad in ({"A": {"eta": float("nan")}}, {"A": {"eta": float("inf")}},
                    {"A": [{1: "x"}]}, {"A": {1, 2}}):
            with self.assertRaises((TypeError, ValueError)):
                tracker.observe(bad)
            self.assertEqual(tracker.observe(raw), first)
        raw["A"]["eta"] = 12
        changed = tracker.observe(raw)
        self.assertGreater(changed, first)
        raw["A"]["eta"] = 3
        self.assertGreater(tracker.observe(raw), changed)

    def test_regression_and_recovery_do_not_replace_the_latest_candidate_map(self):
        for label, valid, blocked in (("구급", "COMMIT", "WAIT"),
                                      ("화재", "OUT_OF_SCOPE", "OUT_OF_SCOPE")):
            with self.subTest(label=label):
                guard = Guard("map-recovery")
                def refresh(version, alternative):
                    guard.advance(1, supports={"dispatch": {label: [1]}}, location=("북문", 1),
                                  vehicles={"A": "available", "B": alternative}, hospitals={},
                                  candidate_versions={"vehicle": version})
                def choose():
                    return guard.apply(guard.begin("vehicle"), reply("vehicle", "A", ("A", "B")))
                refresh(4, "available")
                guard.apply(guard.begin("dispatch"), reply("dispatch", label, DISPATCH))
                self.assertEqual(choose().status, valid)
                refresh(3, "unavailable")
                self.assertEqual(guard.active("vehicle").status, blocked)
                self.assertEqual(choose().status, blocked)
                refresh(4, "available")
                self.assertEqual(guard.active("vehicle").status, blocked)
                self.assertEqual(choose().status, valid)

    def test_conflicting_versions_against_full_event_history(self):
        # Recompute admissibility from the whole input history rather than
        # reproducing the guard's incremental version tracking.
        for seed in range(256):
            rng = random.Random(seed)
            self.guard = Guard("story")
            events = []
            previous = None
            committed = False
            for _ in range(8):
                version = rng.choice((None, 0, 1, 2, 3))
                alternative = rng.choice(("unknown", "available"))
                events.append((version, alternative))
                self.refresh({} if version is None else {"vehicle": version},
                             vehicles={"A": "available", "B": alternative})
                if len(events) == 1:
                    self.apply("dispatch", "구급")
                numbered = [v for v, _ in events if v is not None]
                allowed = not numbered or (version == max(numbered) and
                          len({state for v, state in events if v == version}) == 1)
                retained = committed and allowed and previous == events[-1]
                self.assertEqual(self.guard.active("vehicle").status,
                                 "COMMIT" if retained else "WAIT", (seed, events))
                self.assertEqual(self.apply("vehicle", "A").status,
                                 "COMMIT" if allowed else "WAIT", (seed, events))
                previous, committed = events[-1], allowed
