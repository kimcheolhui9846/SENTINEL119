"""Extend a prepared full-call training set and reserve new comparison and test groups."""

import argparse
import json
from pathlib import Path
import random
from uuid import uuid4

from .prepare import digest, eligible_groups, scan, stratified_take, write_split
from .train import load_split, write_json


def prepare_comparison(root, reference, transcript_fit, output, train_count=8000, reserve_count=1000):
    if output.exists():
        raise ValueError("Use a new output directory")
    saved = {}
    for name in ("train", "dev", "test"):
        examples, parent = load_split(reference, name)
        saved[name] = [e["_meta"]["group_id"] for e in examples]
    if train_count <= len(saved["train"]) or reserve_count <= 0:
        raise ValueError("Request a larger train set and positive reserved split sizes")
    fit = json.loads(transcript_fit.read_text(encoding="utf-8"))
    if fit["split_hashes"] != {k: v["sha256"] for k, v in parent["splits"].items()}:
        raise ValueError("Transcript model was fitted on a different reference dataset")
    entries, groups, report = scan(root)
    pools, audit = eligible_groups(entries, groups)
    seed = parent["seed"]
    dev, rest = stratified_take(pools["Training"], len(saved["dev"]), seed)
    train, remaining = stratified_take(rest, len(saved["train"]), seed + 1)
    old_test, validation = stratified_take(pools["Validation"], len(saved["test"]), seed + 2)
    for name, items in (("dev", dev), ("train", train), ("test", old_test)):
        if [gid for gid, _ in items] != saved[name]:
            raise ValueError(f"Cannot reproduce reference {name} groups from source archives")
    lexical, _ = stratified_take(rest, fit["train_count"], seed + 1)
    lexical_ids = {gid for gid, _ in lexical}
    if (len(lexical_ids) != fit["train_count"]
            or digest("\n".join(sorted(lexical_ids))) != fit["train_groups_sha256"]):
        raise ValueError("Cannot reproduce the transcript model's training groups")
    confirm_pool = [item for item in remaining if item[0] not in lexical_ids]
    confirm, _ = stratified_take(confirm_pool, reserve_count, seed + 3)
    confirm_ids = {gid for gid, _ in confirm}
    extra, _ = stratified_take([item for item in remaining if item[0] not in confirm_ids],
                               train_count - len(train), seed + 4)
    expanded = train + extra
    random.Random(seed + 5).shuffle(expanded)
    test, _ = stratified_take(validation, reserve_count, seed + 6)
    if len(expanded) != train_count or min(len(confirm), len(test)) != reserve_count:
        raise ValueError("Insufficient eligible groups after excluding prior exposure")
    report.update(audit)
    report.update({"schema_version": 1, "seed": seed, "label_scope": "full_call",
                   "parent_splits": parent["splits"], "excluded_transcript_groups_sha256": fit["train_groups_sha256"],
                   "excluded_transcript_count_from_confirm": len(lexical_ids),
                   "group_counts": {k: len(v) for k, v in pools.items()}, "splits": {}})
    output.parent.mkdir(exist_ok=True, parents=True)
    # Use a sibling so the published dataset inherits the same workspace ACLs as its parent.
    staged = output.with_name(f".{output.name}-{uuid4().hex}.partial")
    staged.mkdir()
    try:
        for name, items in (("train", expanded), ("dev", dev), ("confirm", confirm), ("test", test)):
            report["splits"][name] = write_split(staged, name, items)
        if report["splits"]["dev"] != parent["splits"]["dev"]:
            raise ValueError("Reference dev data changed")
        write_json(staged / "manifest.json", report)
        staged.rename(output)
    finally:
        if staged.exists():
            for file in staged.iterdir():
                file.unlink()
            staged.rmdir()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--transcript-fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_comparison(args.data_root, args.reference, args.transcript_fit, args.output)
    print(json.dumps({k: v["calls"] for k, v in report["splits"].items()}))


if __name__ == "__main__":
    main()
