"""Select a full-call policy on dev, or evaluate the frozen dev policy on test."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sentinel119.train import classification_metrics, load_split, write_json
from sentinel119.transcript import LABELS, combine_scores, lexical_scores


def metrics(name, truth, scores):
    labels = LABELS[name]
    predictions = [labels[i] for i in scores.argmax(axis=1)]
    result = classification_metrics(truth, predictions, labels)
    if name == "urgency":
        severity = {"high": 2, "medium": 1, "low": 0}
        result["downward"] = sum(severity[p] < severity[a] for a, p in zip(truth, predictions))
        result["high_to_low"] = sum(a == "high" and p == "low" for a, p in zip(truth, predictions))
    return result


def select(name, truth, pointer, lexical):
    baseline = metrics(name, truth, pointer)
    candidates = []
    for weight in (0, .25, .5, .75):
        for offset in ((0, .25, .5, .75, 1) if name == "urgency" else (0,)):
            config = {"transcript_weight": weight, "severity_offset": offset}
            result = metrics(name, truth, combine_scores(name, pointer, lexical, config))
            if name == "urgency":
                eligible = (result["downward"] <= .8 * baseline["downward"]
                            and result["high_to_low"] <= baseline["high_to_low"]
                            and result["accuracy"] >= baseline["accuracy"])
            else:
                eligible = result["accuracy"] >= baseline["accuracy"] - .01
            candidates.append({"config": config, "metrics": result, "eligible": eligible})
    valid = [c for c in candidates if c["eligible"] and c["metrics"]["macro_f1"] >= baseline["macro_f1"]]
    best = max(valid, key=lambda c: (c["metrics"]["macro_f1"], c["metrics"]["accuracy"])) if valid else candidates[0]
    return best["config"], candidates


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dev-policy", type=Path, help="Required to evaluate on test")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    split = "test" if args.dev_policy else "dev"
    examples, manifest = load_split(args.data, split)
    exported = read_json(args.scores)
    fit = read_json(args.model / "fit.json")
    path = args.model / "transcript.npz"
    if (fit["labels"] != LABELS or hashlib.sha256(path.read_bytes()).hexdigest() != fit["sha256"]
            or fit["split_hashes"] != {k: v["sha256"] for k, v in manifest["splits"].items()}):
        raise ValueError("Transcript weights or data splits do not match the training manifest")
    if exported["split"] != split or exported["data_sha256"] != fit["split_hashes"][split]:
        raise ValueError("Logits and split must match")
    identity = {"model_id": exported["model"]["model_id"], "revision": exported["model"]["revision"],
                "pointer_sha256": exported["checkpoint_sha256"], "transcript_sha256": fit["sha256"],
                "split_hashes": fit["split_hashes"], "labels": LABELS}
    if not identity["pointer_sha256"]:
        raise ValueError("Expected an adapted pointer checkpoint")
    policy = read_json(args.dev_policy) if args.dev_policy else {"identity": identity, "tasks": {}}
    if policy["identity"] != identity:
        raise ValueError("Dev policy requires the same checkpoint, label order and data splits")
    scores = {s["group_id"]: s["logits"] for s in exported["scores"]}
    if (len(scores) != len(exported["scores"]) or len(scores) != len(examples)
            or set(scores) != {e["_meta"]["group_id"] for e in examples}):
        raise ValueError("Require complete one-to-one coverage of the evaluation split")
    with np.load(path, allow_pickle=False) as weights:
        lexical = lexical_scores((e["state"] for e in examples), weights)
    report = {"split": split, "calls": len(examples), "tasks": {}}
    for name, labels in LABELS.items():
        truth = [e["questions"][name]["label"] for e in examples]
        pointer = np.array([[scores[e["_meta"]["group_id"]][name][k] for k in labels] for e in examples])
        if not args.dev_policy:
            policy["tasks"][name], candidates = select(name, truth, pointer, lexical[name])
            report.setdefault("dev_grid", {})[name] = candidates
        config = policy["tasks"][name]
        combined = combine_scores(name, pointer, lexical[name], config)
        report["tasks"][name] = {
            "sft": metrics(name, truth, pointer), "transcript_only": metrics(name, truth, lexical[name]),
            "combined": metrics(name, truth, combined)}
    args.output.mkdir(parents=True)
    if not args.dev_policy:
        write_json(args.output / "policy.json", policy)
    report["policy"] = policy
    write_json(args.output / "metrics.json", report)
    print(json.dumps({k: v["combined"] for k, v in report["tasks"].items()}))


if __name__ == "__main__":
    main()
