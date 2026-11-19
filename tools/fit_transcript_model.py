"""Fit a hashed n-gram model on official Training groups outside the fixed dev set."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer

from sentinel119.prepare import eligible_groups, make_example, scan, stratified_take
from sentinel119.train import load_split, write_json
from sentinel119.transcript import LABELS, counts, tfidf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=40000)
    args = parser.parse_args()
    if args.train_count <= 0 or args.output.exists():
        parser.error("Use a positive training size and a new output directory")
    dev, manifest = load_split(args.data, "dev")
    entries, groups, _ = scan(args.data_root)
    pools, _ = eligible_groups(entries, groups)
    dev_groups, training = stratified_take(pools["Training"], len(dev), manifest["seed"])
    if {gid for gid, _ in dev_groups} != {e["_meta"]["group_id"] for e in dev}:
        raise ValueError("Fixed development groups do not match this archive collection")
    selected, _ = stratified_take(training, args.train_count, manifest["seed"] + 1)
    if len(selected) != args.train_count:
        raise ValueError("Too few eligible training groups")
    states, targets = [], {name: [] for name in LABELS}
    with ExitStack() as stack:
        archives = {}
        for gid, entry in selected:
            if entry.archive not in archives:
                archives[entry.archive] = stack.enter_context(zipfile.ZipFile(entry.archive))
            raw = json.loads(archives[entry.archive].read(entry.member).decode("utf-8-sig"))
            example = make_example(raw, gid, {})
            if (example["_meta"]["record_id"], example["_meta"]["content_hash"],
                tuple(example["questions"][k]["label"] for k in LABELS)) != (
                    entry.record_id, entry.content_hash, entry.labels):
                raise ValueError("Archive changed after split audit")
            states.append(example["state"])
            for name in targets:
                targets[name].append(example["questions"][name]["label"])
    print(f"Vectorizing {len(states)} calls", flush=True)
    matrix = counts(states)
    idf = TfidfTransformer().fit(matrix).idf_
    matrix = tfidf(matrix, idf)
    weights = {"idf": idf}
    for name, labels in LABELS.items():
        coef, intercept = [], []
        for label in labels:
            binary = np.array(targets[name]) == label
            model = LogisticRegression(solver="liblinear", dual=True, C=8, random_state=119)
            model.fit(matrix, binary)
            coef.append(model.coef_[0])
            intercept.append(model.intercept_[0])
        weights[name + "_coef"] = np.array(coef, dtype=np.float32)
        weights[name + "_intercept"] = np.array(intercept, dtype=np.float32)
    args.output.mkdir(parents=True)
    path = args.output / "transcript.npz"
    np.savez_compressed(path, **weights)
    write_json(args.output / "fit.json", {
        "train_count": len(states), "labels": LABELS, "seed": manifest["seed"],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "train_groups_sha256": hashlib.sha256("\n".join(sorted(g for g, _ in selected)).encode()).hexdigest(),
        "split_hashes": {k: v["sha256"] for k, v in manifest["splits"].items()}})
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
