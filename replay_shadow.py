"""Compare three dispatch decisions on the same saved model responses."""

import argparse
import datetime
import json
import statistics
from pathlib import Path

from shadow_harness import POLICY_VERSION, decide_shadow, valid_dispatch_scores


CATEGORIES = {"ems", "fire", "rescue", "other"}
RESOURCE_CATEGORIES = {"ems", "fire"}
POLICIES = ("model_top1", "legacy", "candidate")


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _validate_call(call):
    _check(isinstance(call, dict), "each line must contain a call object")
    for field in ("call_id", "model_id", "adapter_version", "captured_at"):
        _check(_text(call.get(field)), f"{field} must be nonempty text")
    if "legacy_policy_blob" in call:
        _check(_text(call["legacy_policy_blob"]), "legacy_policy_blob must be nonempty text")
    _check(call.get("split") in ("dev", "test"), "split must be dev or test")
    try:
        captured_at = datetime.datetime.fromisoformat(call["captured_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("captured_at must be an ISO timestamp with a timezone") from exc
    _check(captured_at.utcoffset() is not None, "captured_at must be an ISO timestamp with a timezone")
    _check(isinstance(call.get("turns"), list) and call["turns"], "turns must be a nonempty list")


def _validate_turn(turn):
    _check(isinstance(turn, dict), "turn must be an object")
    _check(turn.get("speaker") in ("신고자", "수보요원"), "invalid speaker")
    _check(_text(turn.get("text")), "text must be nonempty")
    _check(valid_dispatch_scores(turn.get("dispatch_scores")), "invalid dispatch_scores")

    eligible_units = turn.get("eligible_units")
    _check(isinstance(eligible_units, dict) and set(eligible_units) <= RESOURCE_CATEGORIES, "invalid eligible_units")
    for units in eligible_units.values():
        _check(isinstance(units, list) and all(map(_text, units)), "invalid eligible_units IDs")
        _check(len(units) == len(set(units)), "duplicate eligible_units IDs")

    for field in ("legacy_category", "gold_category"):
        label = turn.get(field)
        _check(field in turn and (label is None or isinstance(label, str) and label in CATEGORIES), f"invalid {field}")
    for field in ("legacy_proposed_unit", "gold_proposed_unit"):
        _check(field in turn and (turn[field] is None or _text(turn[field])), f"invalid {field}")
    if turn["gold_proposed_unit"] is not None:
        _check(turn["gold_proposed_unit"] in eligible_units.get(turn["gold_category"], ()), "gold_proposed_unit is ineligible")

    applied = turn.get("applied_unit")
    if applied is not None:
        _check(isinstance(applied, dict) and applied.get("category") in ("ems", "fire")
               and _text(applied.get("unit_id")), "invalid applied_unit")
        _check(turn["gold_proposed_unit"] is None, "gold_proposed_unit must be null when applied_unit is set")


def _top_choice(scores):
    best = max(scores.values())
    winners = [category for category, score in scores.items() if score == best]
    return winners[0] if len(winners) == 1 else None


def _replay(call):
    prefix = []
    for turn_number, turn in enumerate(call["turns"], 1):
        try:
            _validate_turn(turn)
        except ValueError as exc:
            raise ValueError(f"turn {turn_number}: {exc}") from exc
        prefix.append({"speaker": turn["speaker"], "text": turn["text"]})
        candidate = decide_shadow(
            prefix.copy(), turn["dispatch_scores"], turn["eligible_units"],
            model_id=call["model_id"], adapter_version=call["adapter_version"],
            applied_unit=turn.get("applied_unit"), subtype_scores=turn.get("subtype_scores"),
        )
        yield {
            "turn": turn_number,
            "gold_category": turn["gold_category"],
            "gold_proposed_unit": turn["gold_proposed_unit"],
            "eligible_units": turn["eligible_units"],
            "already_applied": turn.get("applied_unit") is not None,
            "model_top1": {"category": _top_choice(turn["dispatch_scores"])},
            "legacy": {"category": turn["legacy_category"], "proposed_unit": turn["legacy_proposed_unit"]},
            "candidate": candidate,
        }


def _time_to_correct(calls, policy):
    episodes = 0
    delays = []
    for turns in calls:
        active_category = None
        first_sufficient = None
        for row in turns:
            gold = row["gold_category"]
            if gold != active_category:
                active_category = gold
                first_sufficient = row["turn"] if gold is not None else None
                episodes += gold is not None
            if first_sufficient is not None and row[policy]["category"] == gold:
                delays.append(row["turn"] - first_sufficient)
                first_sufficient = None
    return {
        "episodes": episodes,
        "resolved": len(delays),
        "unresolved": episodes - len(delays),
        "mean_turns": statistics.fmean(delays) if delays else None,
    }


def _category_metrics(rows, calls, policy):
    gold = [row["gold_category"] for row in rows]
    predictions = [row[policy]["category"] for row in rows]
    sufficient = sum(label is not None for label in gold)
    commits = sum(label is not None for label in predictions)
    correct = sum(prediction is not None and prediction == target for prediction, target in zip(predictions, gold))
    wrong = commits - correct
    return {
        "turns": len(rows),
        "sufficient_turns": sufficient,
        "commits": commits,
        "correct_commits": correct,
        "wrong_commits": wrong,
        "unnecessary_waits": sum(prediction is None and target is not None for prediction, target in zip(predictions, gold)),
        "coverage": commits / len(rows),
        "wrong_commit_rate": wrong / commits if commits else None,
        "time_to_correct": _time_to_correct(calls, policy),
    }


def _unit_metrics(rows, policy):
    proposals = wrong = ineligible = missed = 0
    for row in rows:
        target = row["gold_proposed_unit"]
        decision = row[policy]
        unit_id = decision["proposed_unit"]
        if unit_id is None:
            missed += target is not None
            continue
        proposals += 1
        eligible = not row["already_applied"] and unit_id in row["eligible_units"].get(decision["category"], ())
        ineligible += not eligible
        wrong += not eligible or unit_id != target or decision["category"] != row["gold_category"]
    return {
        "turns": len(rows),
        "expected_proposals": sum(row["gold_proposed_unit"] is not None for row in rows),
        "proposals": proposals,
        "wrong_proposals": wrong,
        "ineligible_proposals": ineligible,
        "missed_proposals": missed,
    }


def compare_calls(records, *, details=False):
    """Replay curated per-call records against prefix labels and observed legacy output."""
    by_split = {"dev": [], "test": []}
    seen_ids = set()
    identity = None
    legacy_policy_blob = None
    captures = []
    for line, call in enumerate(records, 1):
        try:
            _validate_call(call)
            call_id = call["call_id"]
            _check(call_id not in seen_ids, "duplicate call_id")
            seen_ids.add(call_id)
            call_identity = (call["model_id"], call["adapter_version"])
            if identity is None:
                identity = call_identity
                legacy_policy_blob = call.get("legacy_policy_blob")
            _check(call_identity == identity, "mixed model_id or adapter_version")
            _check(call.get("legacy_policy_blob") == legacy_policy_blob, "mixed legacy_policy_blob provenance")
            turns = list(_replay(call))
        except ValueError as exc:
            raise ValueError(f"call record {line}: {exc}") from exc
        by_split[call["split"]].append(turns)
        if details:
            captures.append({"call_id": call_id, "split": call["split"], "captured_at": call["captured_at"], "turns": turns})

    _check(identity is not None, "at least one call record is required")
    report = {
        "model_id": identity[0],
        "adapter_version": identity[1],
        "policy_version": POLICY_VERSION,
        "legacy_policy_blob": legacy_policy_blob,
        "policies": {
            "model_top1": "unique model argmax for category comparison",
            "legacy": "recorded category and new unit decision from the existing policy",
            "candidate": "shadow category and suggested unit without applying a dispatch",
        },
        "splits": {},
    }
    for split, calls in by_split.items():
        if not calls:
            continue
        rows = [row for turns in calls for row in turns]
        report["splits"][split] = {
            "calls": len(calls),
            "category": {policy: _category_metrics(rows, calls, policy) for policy in POLICIES},
            "unit_proposals": {policy: _unit_metrics(rows, policy) for policy in ("legacy", "candidate")},
        }
    if details:
        report["details"] = captures
    return report


def _read_calls(path):
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_number}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="UTF-8 JSONL, one captured call per line")
    parser.add_argument("--details", action="store_true", help="include per-turn decisions and candidate reasons")
    args = parser.parse_args()
    try:
        report = compare_calls(_read_calls(args.records), details=args.details)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
