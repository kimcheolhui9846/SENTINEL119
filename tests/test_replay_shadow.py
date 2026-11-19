import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from replay_shadow import compare_calls
from shadow_harness import POLICY_VERSION


REQUEST = "성수동에서 사람이 갑자기 쓰러졌어요. 구급차를 보내 주세요."
EMS_SCORES = {"ems": 0.97, "fire": 0.01, "rescue": 0.01, "other": 0.01}
FIRE_SCORES = {"ems": 0.01, "fire": 0.97, "rescue": 0.01, "other": 0.01}
TIED_SCORES = {"ems": 0.5, "fire": 0.5, "rescue": 0, "other": 0}


def turn(**overrides):
    return {
        "speaker": "신고자",
        "text": REQUEST,
        "dispatch_scores": EMS_SCORES.copy(),
        "eligible_units": {"ems": ["A2"], "fire": ["F1"]},
        "legacy_category": None,
        "legacy_proposed_unit": None,
        "gold_category": "ems",
        "gold_proposed_unit": "A2",
        **overrides,
    }


def call(*turns, **overrides):
    return {
        "call_id": "sample-1",
        "split": "test",
        "model_id": "checkpoint-a",
        "adapter_version": "adapter-v1",
        "captured_at": "2026-11-20T00:00:00Z",
        "turns": list(turns),
        **overrides,
    }


class ReplayShadowTests(unittest.TestCase):
    def test_policy_receives_the_current_response_and_prefix_only(self):
        first = turn(scenario="fire", future_decision="emu", legacy_category="fire")
        second = turn(text="미래 발화", dispatch_scores=FIRE_SCORES, gold_category="fire", gold_proposed_unit="F1")
        empty_decision = {"category": None, "proposed_unit": None, "needs_review": False}
        with patch("replay_shadow.decide_shadow", return_value=empty_decision) as decide:
            compare_calls([call(first, second, scenario="fire")])

        first_args, first_kwargs = decide.call_args_list[0]
        self.assertEqual(first_args, ([{"speaker": "신고자", "text": REQUEST}], EMS_SCORES, first["eligible_units"]))
        self.assertEqual(first_kwargs, {
            "model_id": "checkpoint-a", "adapter_version": "adapter-v1", "applied_unit": None, "subtype_scores": None,
        })
        self.assertEqual(decide.call_args_list[1].args[0][-1], {"speaker": "신고자", "text": "미래 발화"})

    def test_tie_and_all_wait_preserve_zero_coverage(self):
        record = call(turn(dispatch_scores=TIED_SCORES))
        policies = compare_calls([record])["splits"]["test"]["category"]
        for policy in policies.values():
            self.assertEqual(policy["coverage"], 0.0)
            self.assertIsNone(policy["wrong_commit_rate"])
            self.assertEqual(policy["unnecessary_waits"], 1)
            self.assertEqual(policy["time_to_correct"]["unresolved"], 1)

    def test_premature_match_and_later_label_do_not_distort_turn_delay(self):
        record = call(
            turn(gold_category=None, gold_proposed_unit=None),
            turn(dispatch_scores=FIRE_SCORES),
            turn(dispatch_scores=EMS_SCORES, gold_category="fire", gold_proposed_unit="F1"),
            turn(dispatch_scores=FIRE_SCORES, gold_category="fire", gold_proposed_unit="F1"),
        )
        metrics = compare_calls([record])["splits"]["test"]["category"]["model_top1"]
        self.assertEqual(metrics["wrong_commits"], 3)
        self.assertEqual(metrics["time_to_correct"], {"episodes": 2, "resolved": 1, "unresolved": 1, "mean_turns": 1.0})

    def test_legacy_proposal_uses_the_matching_eligible_category(self):
        record = call(turn(legacy_category="fire", legacy_proposed_unit="A2"))
        proposals = compare_calls([record])["splits"]["test"]["unit_proposals"]
        self.assertEqual(proposals["legacy"]["ineligible_proposals"], 1)
        self.assertEqual(proposals["legacy"]["wrong_proposals"], 1)

    def test_duplicate_ids_and_mixed_capture_identity_are_rejected(self):
        self.assertRaisesRegex(ValueError, "duplicate call_id", compare_calls, [call(turn()), call(turn(), split="dev")])
        for change in ({"model_id": "checkpoint-b"}, {"adapter_version": "adapter-v2"}):
            with self.subTest(change=change):
                self.assertRaisesRegex(
                    ValueError, "mixed model_id or adapter_version", compare_calls,
                    [call(turn()), call(turn(), call_id="sample-2", split="dev", **change)],
                )

    def test_legacy_policy_blob_provenance_is_consistent_and_summarized(self):
        first = call(turn(), legacy_policy_blob="a" * 40)
        second = call(turn(), call_id="sample-2", split="dev", legacy_policy_blob="a" * 40)
        summary = compare_calls([first, second])
        self.assertEqual(summary["legacy_policy_blob"], "a" * 40)
        self.assertEqual(summary["policy_version"], POLICY_VERSION)
        self.assertIsNone(compare_calls([call(turn())])["legacy_policy_blob"])
        for change in ({"legacy_policy_blob": "b" * 40}, {}):
            with self.subTest(change=change):
                self.assertRaisesRegex(
                    ValueError, "mixed legacy_policy_blob", compare_calls,
                    [first, call(turn(), call_id="sample-2", split="dev", **change)],
                )

    def test_gold_cannot_require_new_proposal_when_a_unit_was_applied(self):
        record = call(turn(applied_unit={"category": "ems", "unit_id": "A1"}))
        self.assertRaisesRegex(ValueError, "gold_proposed_unit.*applied_unit", compare_calls, [record])

    def test_invalid_required_fields_fail_at_the_input_boundary(self):
        for field, value in (
            ("dispatch_scores", {**EMS_SCORES, "ems": True}),
            ("gold_category", "unknown"),
            ("eligible_units", {"ems": ["A2", "A2"], "fire": []}),
        ):
            with self.subTest(field=field):
                self.assertRaisesRegex(ValueError, field, compare_calls, [call(turn(**{field: value}))])

    def test_cli_replays_jsonl_and_stdout_contains_json_only(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".jsonl", dir=repository, delete=False) as stream:
            input_path = Path(stream.name)
            stream.write(json.dumps(call(turn()), ensure_ascii=False) + "\n")
        try:
            result = subprocess.run(
                [sys.executable, "-B", "-m", "replay_shadow", str(input_path), "--details"],
                capture_output=True, text=True, encoding="utf-8", check=True,
                cwd=repository,
            )
        finally:
            input_path.unlink()

        report = json.loads(result.stdout)
        self.assertEqual(result.stderr, "")
        self.assertEqual(report["splits"]["test"]["category"]["candidate"]["correct_commits"], 1)
        self.assertEqual(report["details"][0]["turns"][0]["candidate"]["evidence_turn"], 1)
