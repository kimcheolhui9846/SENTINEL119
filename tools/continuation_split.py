"""Build a continuation dataset: the training calls a checkpoint has not seen yet, in the original epoch order.

`sentinel119.train.fit` draws its first epoch as random.Random(seed).shuffle(range(len(train))) before any
other use of that generator (no oversampling). After `consumed` calls the model has seen order[:consumed];
this tool writes order[consumed:] as the new train split and copies dev/confirm/test unchanged, so a run
started with --init-checkpoint continues on unseen calls and stays comparable on the same evaluation calls.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil

from sentinel119.train import load_split, write_json


def continuation(data, output, seed, consumed):
    if output.exists():
        raise ValueError("Use a new output directory")
    train, manifest = load_split(data, "train")
    if not 0 < consumed < len(train):
        raise ValueError(f"consumed must be between 1 and {len(train) - 1}")
    order = list(range(len(train)))
    random.Random(seed).shuffle(order)
    rest = [train[i] for i in order[consumed:]]
    output.mkdir(parents=True)
    with (output / "train.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for example in rest:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    splits = dict(manifest["splits"])
    splits["train"] = {**splits["train"], "calls": len(rest),
                       "sha256": hashlib.sha256((output / "train.jsonl").read_bytes()).hexdigest()}
    for name in splits:
        if name != "train":
            shutil.copyfile(data / f"{name}.jsonl", output / f"{name}.jsonl")
    write_json(output / "manifest.json", {**manifest, "splits": splits, "continuation": {
        "parent": data.name, "parent_train_sha256": manifest["splits"]["train"]["sha256"],
        "seed": seed, "consumed_calls": consumed, "remaining_calls": len(rest)}})
    return len(rest)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="Seed of the run being continued")
    parser.add_argument("--consumed", type=int, required=True,
                        help="Calls the checkpoint has seen: selected step x grad-accum x calls-per-batch")
    args = parser.parse_args()
    print(continuation(args.data, args.output, args.seed, args.consumed))


if __name__ == "__main__":
    main()
