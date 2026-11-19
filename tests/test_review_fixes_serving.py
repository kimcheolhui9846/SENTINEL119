"""Regression tests for the second review: overflow-aware tools, run-continuation guards, staged augmentation."""
import importlib.util
import json
from pathlib import Path
import random
import sys

import pytest

from sentinel119.auxiliary import augment
from sentinel119.decision import read_examples
from sentinel119.prepare import prepare
from sentinel119.train import main as train_main
from test_auxiliary import source as aux_source
from test_continuation import cs, hier
from test_review_fixes import dr, split_and_scores


def test_decision_rule_accepts_recorded_overflow_calls(tmp_path):
    data, scores = split_and_scores(tmp_path, ids=("a", "b", "c"), score_ids=["a", "c"])
    payload = json.loads(scores.read_text())
    with pytest.raises(ValueError, match="missing"):
        dr.load(data, "confirm", scores)
    payload["coverage"] = {"overflow_group_ids": ["b"]}
    scores.write_text(json.dumps(payload))
    loaded = dr.load(data, "confirm", scores)
    assert loaded["ids"] == ["a", "c"] and loaded["overflow_group_ids"] == ["b"]


def fake_run(path, data, seed=119, overflow=(), oversample=None):
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    path.mkdir()
    (path / "run.json").write_text(json.dumps({
        "status": "completed", "config": {"seed": seed, "oversample": oversample},
        "dataset_manifest": manifest,
        "result": {"coverage": {"train": {"overflow_group_ids": list(overflow)}}}}), encoding="utf-8")
    return path


def test_continuation_rebuilds_the_order_without_overflow_calls(tmp_path):
    data = hier(tmp_path)
    train = read_examples(data / "train.jsonl")
    dropped = train[3]["_meta"]["group_id"]
    run = fake_run(tmp_path / "run", data, overflow=[dropped])
    out = tmp_path / "cont"
    assert cs.continuation(data, out, seed=119, consumed=5, run=run) == 6
    used = [e for e in train if e["_meta"]["group_id"] != dropped]
    order = list(range(len(used))); random.Random(119).shuffle(order)
    rest = read_examples(out / "train.jsonl")
    assert [e["_meta"]["group_id"] for e in rest] == [used[i]["_meta"]["group_id"] for i in order[5:]]
    with pytest.raises(ValueError, match="seed"):
        cs.continuation(data, tmp_path / "bad", seed=7, consumed=5, run=run)


def test_failed_augmentation_leaves_no_output(tmp_path):
    root = aux_source(tmp_path)
    parent, child = tmp_path / "parent", tmp_path / "child"
    prepare(root, parent, train_count=3, dev_count=2, test_count=2)
    prepare(root, child, train_count=4, dev_count=2, test_count=2)
    manifest = json.loads((child / "manifest.json").read_text(encoding="utf-8"))
    manifest["parent_splits"] = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))["splits"]
    (child / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "child-aug"
    with pytest.raises(ValueError, match="--parent"):
        augment(root, child, out)
    assert not out.exists() and not out.with_name("child-aug.partial").exists()
    augment(root, parent, tmp_path / "parent-aug")
    augment(root, child, out, parent=tmp_path / "parent-aug")
    assert (out / "manifest.json").exists()


def test_composite_selection_needs_both_primary_questions(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "--data", str(tmp_path), "--output", str(tmp_path / "o"),
                                      "--select-by", "composite", "--questions", "dispatch,ems_subtype"])
    with pytest.raises(SystemExit):
        train_main()


def test_init_checkpoint_refuses_changed_lora_settings(tmp_path, monkeypatch):
    init = tmp_path / "init"
    init.mkdir()
    (init / "run.json").write_text(json.dumps({"config": {"lora_rank": 16, "lora_alpha": 32,
                                                          "lora_targets": "attention"}}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train", "--data", str(tmp_path), "--output", str(tmp_path / "o"),
                                      "--mode", "lora", "--init-checkpoint", str(init), "--lora-rank", "4"])
    with pytest.raises(SystemExit):
        train_main()
