"""Regression tests for review findings: JSONL separators, schedule, optional dispatch, policy guards."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

import pytest

from sentinel119.decision import read_examples
from sentinel119.prepare import make_example
from sentinel119.train import checkpoint_questions, learning_rate_scale, training_order
from test_prepare import call

spec = importlib.util.spec_from_file_location("decision_rule", Path(__file__).parents[1] / "tools" / "decision_rule.py")
dr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dr)


def test_jsonl_record_with_unicode_line_separator_loads(tmp_path):
    example = make_example(call(text="숨을 안 쉬어요"), "group", {})
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(example, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    loaded = read_examples(path)
    assert len(loaded) == 1 and " " in loaded[0]["state"]


def test_cosine_schedule_never_starts_or_ends_at_zero():
    assert learning_rate_scale(1, 1, 0.0, "cosine") == 1.0
    assert all(learning_rate_scale(s, 10, 0.0, "cosine") > 0 for s in range(1, 11))


def test_training_order_without_dispatch_question():
    urgency_only = [{"questions": {"urgency": {"label": "high"}}}] * 3
    assert sorted(training_order(urgency_only, {}, random.Random(0))) == [0, 1, 2]
    with pytest.raises(ValueError, match="dispatch"):
        training_order(urgency_only, {"other": 3}, random.Random(0))


def test_checkpoint_questions_follow_request_then_checkpoint():
    prior = {"config": {"questions": "dispatch"}}
    assert checkpoint_questions(None, prior) == {"dispatch"}
    assert checkpoint_questions("dispatch,urgency", prior) == {"dispatch", "urgency"}
    assert checkpoint_questions(None, None) is None


def split_and_scores(tmp_path, ids=("a", "b"), score_ids=None, split="confirm", model="m1"):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    rows = [{"_meta": {"group_id": g}, "questions": {"dispatch": {"label": "ems"}, "urgency": {"label": "high"}}} for g in ids]
    payload = "".join(json.dumps(r) + "\n" for r in rows).encode()
    (data / f"{split}.jsonl").write_bytes(payload)
    logits = {"dispatch": {"fire": 0, "rescue": 0, "ems": 2, "other": 0}, "urgency": {"high": 1, "medium": 0, "low": 0}}
    scores = {"split": split, "data_sha256": hashlib.sha256(payload).hexdigest(), "model": {"model_id": model},
              "checkpoint_sha256": {"decision_head.pt": model}, "prompt": {"variant": "bare"}, "questions": None,
              "scores": [{"group_id": g, "logits": logits} for g in (score_ids or ids)]}
    path = tmp_path / f"scores-{split}-{model}.json"
    path.write_text(json.dumps(scores))
    return data, path


def test_scores_must_match_the_split_exactly(tmp_path):
    data, ok = split_and_scores(tmp_path)
    assert dr.load(data, "confirm", ok)["ids"] == ["a", "b"]
    _, dup = split_and_scores(tmp_path, score_ids=["a", "a"], model="dup")
    with pytest.raises(ValueError, match="duplicate"):
        dr.load(data, "confirm", dup)
    _, missing = split_and_scores(tmp_path, score_ids=["a"], model="miss")
    with pytest.raises(ValueError, match="missing"):
        dr.load(data, "confirm", missing)
    scores = json.loads(ok.read_text()); scores["split"] = "dev"; ok.write_text(json.dumps(scores))
    with pytest.raises(ValueError, match="split"):
        dr.load(data, "confirm", ok)


def test_cli_refuses_test_tuning_and_foreign_policies(tmp_path, monkeypatch):
    data, conf = split_and_scores(tmp_path, split="confirm", model="m1")
    _, test_m1 = split_and_scores(tmp_path, split="test", model="m1")
    _, test_m2 = split_and_scores(tmp_path, split="test", model="m2")
    monkeypatch.setattr(sys, "argv", ["dr", "--data", str(data), "--split", "test", "--scores", str(test_m1),
                                      "--output", str(tmp_path / "t.json")])
    with pytest.raises(SystemExit):
        dr.main()
    policy = tmp_path / "policy.json"
    monkeypatch.setattr(sys, "argv", ["dr", "--data", str(data), "--split", "confirm", "--scores", str(conf),
                                      "--output", str(policy)])
    dr.main()
    monkeypatch.setattr(sys, "argv", ["dr", "--data", str(data), "--split", "test", "--scores", str(test_m2),
                                      "--apply", str(policy), "--output", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit):
        dr.main()
    monkeypatch.setattr(sys, "argv", ["dr", "--data", str(data), "--split", "test", "--scores", str(test_m1),
                                      "--apply", str(policy), "--output", str(tmp_path / "ok.json")])
    dr.main()
    assert json.loads((tmp_path / "ok.json").read_text())["tasks"]["urgency"]["applied"]
