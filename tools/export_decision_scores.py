"""Export keyed full-call decision logits without transcripts, for offline model comparisons."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from sentinel119.decision import encode
from sentinel119.model import load_zoro, same_base
from sentinel119.prompts import VARIANTS, apply as apply_prompt
from sentinel119.train import evaluation_split, filter_examples, resolve_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "confirm", "test"], required=True)
    parser.add_argument("--reference-data", type=Path, help="Original dataset for a parent checkpoint")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-variant", choices=VARIANTS, help="Base only; checkpoints use their recorded prompt")
    parser.add_argument("--guideline-dir", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output file")
    kwargs = {}
    checkpoint_hashes = {}
    prior = None
    max_length = 4096
    if args.checkpoint:
        prior = json.loads((args.checkpoint / "run.json").read_text(encoding="utf-8"))
        if prior["status"] != "completed":
            raise ValueError("Checkpoint did not complete successfully")
        max_length = prior["config"]["max_length"]
        kwargs["head_path"] = args.checkpoint / "decision_head.pt"
        files = [kwargs["head_path"]]
        if prior["mode"] == "lora":
            kwargs["adapter_path"] = args.checkpoint / "adapter"
            files.extend(kwargs["adapter_path"] / name for name in ("adapter_config.json", "adapter_model.safetensors"))
        checkpoint_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    examples, manifest = evaluation_split(args.data, args.split, prior, args.reference_data)
    prompt = resolve_prompt(args.prompt_variant, args.guideline_dir, prior)
    examples = [apply_prompt(e, prompt) for e in examples]
    model, tokenizer, identity = load_zoro(local_files_only=True, **kwargs)
    if args.checkpoint and not same_base(prior["model"], identity):
        raise ValueError("Base model identity mismatch")
    examples, coverage = filter_examples(examples, tokenizer, max_length)
    scores = []
    model.eval()
    with torch.no_grad():
        for example in examples:
            rows = encode(example, tokenizer, max_length)
            logits = model(rows)
            scores.append({"group_id": example["_meta"]["group_id"], "logits": {
                row["qid"]: dict(zip(row["keys"], z.float().tolist(), strict=True))
                for row, z in zip(rows, logits, strict=True)}})
    result = {"model": identity, "checkpoint_sha256": checkpoint_hashes, "coverage": coverage, "prompt": prompt,
              "split": args.split, "data_sha256": manifest["splits"][args.split]["sha256"], "scores": scores}
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, allow_nan=False)
    print(f"{args.split}: {len(scores)} scored")


if __name__ == "__main__":
    main()
