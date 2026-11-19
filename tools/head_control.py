"""Preflight, run or compare the head-only control against the original 2k LoRA run."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from sentinel119.decision import MODEL_ID, MODEL_REVISION
from sentinel119.train import load_split, write_json


MATCHED = {"max_steps": 500, "grad_accum": 4, "eval_every": 100, "max_length": 4096,
           "learning_rate": 1e-4, "seed": 119}
DEFAULTS = {"warmup_ratio": 0, "schedule": "constant", "select_by": "nll", "oversample": "",
            "urgency_weight": 1, "urgency_cost_weight": 0, "urgency_label_smoothing": 0,
            "dispatch_logit_adjust": 0}


def read_run(directory, mode):
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    cfg, identity = run["config"], run["model"]
    if run["status"] != "completed" or run["mode"] != mode or cfg.get("evaluate_only"):
        raise ValueError(f"Expected a completed {mode} training run")
    if (identity["model_id"] != MODEL_ID or identity["revision"] != MODEL_REVISION or identity["local_override"]
            or run.get("prompt", {"variant": "bare"}).get("variant") != "bare"
            or run.get("prompt", {}).get("overrides") or cfg.get("head_learning_rate") not in (None, 1e-4)):
        raise ValueError("The control requires pinned Zoro, the bare prompt and a matched head LR")
    for k, value in MATCHED.items():
        if cfg[k] != value:
            raise ValueError(f"Unmatched {k}")
    for k, value in DEFAULTS.items():
        if cfg.get(k, value) != value:
            raise ValueError(f"Unmatched {k}")
    if mode == "lora" and (cfg["lora_rank"], cfg["lora_alpha"], cfg.get("lora_targets", "attention-qv")) != (
            8, 16, "attention-qv"):
        raise ValueError("Expected the original q/v rank-8 LoRA baseline")
    if mode == "head-only" and set(run["trainable_names"]) != {"head.q.weight", "head.q.bias",
                                                                "head.k.weight", "head.k.bias"}:
        raise ValueError("Unexpected trainable parameters in the head-only run")
    result = run["result"]
    if not all(result[k] for k in ("base_weights_frozen", "nonzero_head_gradient", "reload_verified")):
        raise ValueError("Failed gradient, frozen-backbone or reload checks")
    if result["nonzero_adapter_gradient"] != (mode == "lora"):
        raise ValueError("Unexpected LoRA gradient status")
    return run


def measures(metrics):
    u, d = metrics["urgency"], metrics["dispatch"]
    high, low = (u["labels"].index(k) for k in ("high", "low"))
    other = d["labels"].index("other")
    return {"dispatch_macro_f1": d["macro_f1"], "urgency_macro_f1": u["macro_f1"],
            "downward_rate": u["downward_error_rate"], "high_to_low": u["confusion_matrix"][high][low],
            "other_correct": d["confusion_matrix"][other][other], "other_support": d["support"]["other"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action", choices=["check", "train", "compare"], default="check")
    args = parser.parse_args()
    baseline = read_run(args.baseline, "lora")
    for name, n in (("train", 2000), ("dev", 300)):
        rows, manifest = load_split(args.data, name)
        if len(rows) != n or manifest["splits"] != baseline["dataset_manifest"]["splits"]:
            raise ValueError("The control requires exactly the original hashed splits")
    flags = [s for k, v in MATCHED.items() for s in ("--" + k.replace("_", "-"), str(v))]
    command = [sys.executable, "-m", "sentinel119.train", "--data", str(args.data), "--output",
               str(args.output), "--mode", "head-only", "--prompt-variant", "bare", "--keep-checkpoints",
               "--local-files-only", *flags]
    if args.action != "compare" and args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a fresh head-only output directory")
    base_metrics = measures(baseline["result"]["adapted_raw_dev"])
    if args.action == "check":
        print(json.dumps({"status": "ready", "verified_calls": {"train": 2000, "dev": 300},
                          "lora_dev": base_metrics, "training_argv": command}, indent=2))
        return
    if args.action == "train":
        subprocess.run(command, check=True)
    head = read_run(args.output, "head-only")
    for k in ("base_head_sha256", "config_sha256", "dtype"):
        if head["model"][k] != baseline["model"][k]:
            raise ValueError(f"Initial model mismatch in {k}")
    if (head["dataset_manifest"]["splits"] != baseline["dataset_manifest"]["splits"]
            or head["result"]["coverage"] != baseline["result"]["coverage"]):
        raise ValueError("Runs used different data or coverage")
    initial_matches = {}
    for task in ("dispatch", "urgency"):
        original = baseline["result"]["base_raw_dev"][task]
        before_fit = head["result"]["base_raw_dev"][task]
        initial_matches[task] = all(before_fit[k] == original[k] for k in ("labels", "confusion_matrix"))
    current = measures(head["result"]["adapted_raw_dev"])
    report = {"split": "dev", "calls": 300, "selection": "minimum dev NLL in each run",
              "pre_training_confusion_matches": initial_matches,
              "best_step": {"lora": baseline["result"]["best_step"], "head_only": head["result"]["best_step"]},
              "lora": base_metrics, "head_only": current,
              "head_minus_lora": {k: current[k] - v for k, v in base_metrics.items() if k != "other_support"}}
    write_json(args.output / "head-comparison.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
