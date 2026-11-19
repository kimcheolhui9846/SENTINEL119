"""Train/evaluate the released Zoro decision head on full-call 119 labels."""

import argparse
import hashlib
from collections import Counter
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

from .decision import ContextOverflow, MAX_BRANCH, UPSTREAM_REVISION, encode, read_examples
from .prompts import VARIANTS, apply as apply_prompt, render as render_prompt


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def classification_metrics(truth, predictions, labels):
    matrix = [[0 for _ in labels] for _ in labels]
    for actual, predicted in zip(truth, predictions):
        matrix[labels.index(actual)][labels.index(predicted)] += 1
    f1 = []
    for i in range(len(labels)):
        tp = matrix[i][i]
        denominator = sum(matrix[i]) + sum(row[i] for row in matrix)
        f1.append(2 * tp / denominator if denominator else 0.0)
    return {"count": len(truth), "accuracy": sum(a == p for a, p in zip(truth, predictions)) / len(truth),
            "macro_f1": sum(f1) / len(labels), "labels": labels, "confusion_matrix": matrix,
            "support": dict(Counter(truth)), "missing_labels": sorted(set(labels) - set(truth))}


def load_split(data_dir, split):
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    examples = read_examples(data_dir / f"{split}.jsonl", manifest["splits"][split]["sha256"])
    expected = "Validation" if split == "test" else "Training"
    for example in examples:
        if example["_meta"]["source"]["official_split"] != expected:
            raise ValueError("Official dataset split does not match requested use")
    return examples, manifest


def evaluation_split(data, split, prior=None, reference_data=None):
    examples, manifest = load_split(data, split)
    if reference_data is None:
        if prior and prior["dataset_manifest"]["splits"] != manifest["splits"]:
            raise ValueError("Evaluation must use the checkpoint's dataset split manifest")
        return examples, manifest
    if (not prior or split not in ("confirm", "test")
            or manifest.get("parent_splits") != prior["dataset_manifest"]["splits"]):
        raise ValueError("Reserved evaluation requires a checkpoint from the named parent splits")
    fields = ("group_id", "record_id", "content_hash")
    reserved = {k: {e["_meta"][k] for e in examples} for k in fields}
    if any(len(ids) != len(examples) for ids in reserved.values()):
        raise ValueError("Duplicate calls in reserved evaluation split")
    for name in ("train", "dev", "test"):
        parent_rows, parent = load_split(reference_data, name)
        if parent["splits"] != manifest["parent_splits"]:
            raise ValueError("Reference data does not match the checkpoint's original splits")
        for k in fields:
            if any(e["_meta"][k] in reserved[k] for e in parent_rows):
                raise ValueError(f"Reserved evaluation overlaps parent {name} on {k}")
    return examples, manifest


def filter_examples(examples, tokenizer, max_length):
    kept, rejected = [], []
    for example in examples:
        try:
            encode(example, tokenizer, max_length)
        except ContextOverflow:
            rejected.append(example["_meta"]["group_id"])
        else:
            kept.append(example)
    if not kept:
        raise ValueError("All calls exceed max_length; increase the budget without truncating full-call labels")
    # Questions may be optional per call (e.g. subtypes only under their dispatch type).
    criteria = {}
    for e in kept:
        for name, q in e["questions"].items():
            criteria.setdefault(name, q["criteria"])
    counts = {name: dict(Counter(e["questions"][name]["label"] for e in kept if name in e["questions"]))
              for name in criteria}
    missing = {name: sorted(set(c) - set(counts[name])) for name, c in criteria.items()}
    return kept, {"input_calls": len(examples), "used_calls": len(kept), "overflow_group_ids": rejected,
                  "used_labels": counts, "missing_labels": missing}


PRIMARY = ("dispatch", "urgency")


def subtype_dispatch(keys, probabilities):
    from .auxiliary import SUBTYPE_DISPATCH
    totals = Counter()
    for key, p in zip(keys, probabilities):
        totals[SUBTYPE_DISPATCH[key]] += p
    return max(totals, key=totals.get)


# Hierarchy: the subtype question that belongs under each dispatch type.
SUBTYPE_OF = {"ems": "ems_subtype", "rescue": "rescue_subtype"}


def evaluate(model, examples, tokenizer, max_length, temperature=1.0, calls_per_batch=1):
    import torch
    from .model import supervised_loss
    actual, predicted, labels = {}, {}, {}
    nll, aux_nll, derived = [], [], []
    hierarchy = Counter()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), calls_per_batch):
            chunk = [encode(e, tokenizer, max_length) for e in examples[start:start + calls_per_batch]]
            flat = [row for rows in chunk for row in rows]
            all_logits = tuple(z / temperature for z in model(flat))
            offset = 0
            for rows in chunk:
                logits = all_logits[offset:offset + len(rows)]
                offset += len(rows)
                # mean_nll stays dispatch+urgency only, so selection matches checkpoints without extra questions.
                main = [(z, r) for z, r in zip(logits, rows) if r["qid"] in PRIMARY]
                aux = [(z, r) for z, r in zip(logits, rows) if r["qid"] not in PRIMARY]
                nll.append(float(supervised_loss(*zip(*main))) if main else float(supervised_loss(logits, rows)))
                if aux:
                    aux_nll.append(float(supervised_loss(*zip(*aux))))
                call = {}
                for row, z in zip(rows, logits):
                    name = row["qid"]
                    labels.setdefault(name, list(row["keys"]))
                    gold, guess = row["keys"][row["label"]], row["keys"][int(z.argmax())]
                    call[name] = (gold, guess)
                    actual.setdefault(name, []).append(gold)
                    predicted.setdefault(name, []).append(guess)
                    if name == "subtype":
                        derived.append(subtype_dispatch(row["keys"], torch.softmax(z.float(), -1).tolist()))
                if "dispatch" in call and call["dispatch"][0] in SUBTYPE_OF and SUBTYPE_OF[call["dispatch"][0]] in call:
                    # End to end: right dispatch type and, under it, the right subtype.
                    gold_d, guess_d = call["dispatch"]
                    gold_s, guess_s = call[SUBTYPE_OF[gold_d]]
                    hierarchy[f"{gold_d}_calls"] += 1
                    hierarchy[f"{gold_d}_end_to_end_correct"] += guess_d == gold_d and guess_s == gold_s
    result = {name: classification_metrics(actual[name], predicted[name], labels[name]) for name in labels}
    for kind in SUBTYPE_OF:
        if hierarchy[f"{kind}_calls"]:
            result[f"{kind}_subtype_end_to_end_accuracy"] = (
                hierarchy[f"{kind}_end_to_end_correct"] / hierarchy[f"{kind}_calls"])
    if derived and "dispatch" in labels:
        # Dispatch implied by summing subtype probabilities within each dispatch type.
        result["dispatch_from_subtype"] = classification_metrics(actual["dispatch"], derived, labels["dispatch"])
    if aux_nll:
        result["auxiliary_mean_nll"] = sum(aux_nll) / len(aux_nll)
    severity = {"low": 0, "medium": 1, "high": 2}
    if "urgency" in actual:
        result["urgency"]["downward_error_rate"] = sum(
            severity[p] < severity[a] for a, p in zip(actual["urgency"], predicted["urgency"])
        ) / len(actual["urgency"])
    result["mean_nll"] = sum(nll) / len(nll)
    result["temperature"] = temperature
    return result


def selection_key(metrics, select_by="nll", max_downward_rate=None):
    """Larger is better. Composite = mean of the two macro-F1s; feasibility (downward cap) ranks first."""
    if select_by == "nll":
        return (True, -metrics["mean_nll"])
    if select_by != "composite":
        raise ValueError(f"Unknown selection rule {select_by}")
    feasible = max_downward_rate is None or metrics["urgency"]["downward_error_rate"] <= max_downward_rate
    return (feasible, (metrics["dispatch"]["macro_f1"] + metrics["urgency"]["macro_f1"]) / 2)


def training_order(train, oversample, rng):
    """Index pool for one epoch; dispatch classes in ``oversample`` are repeated that many times."""
    order = []
    for i, example in enumerate(train):
        repeat = 1
        if oversample:  # only needs the dispatch label when oversampling is requested
            if "dispatch" not in example["questions"]:
                raise ValueError("--oversample needs the dispatch question in the training data")
            repeat = int(oversample.get(example["questions"]["dispatch"]["label"], 1))
        order.extend([i] * repeat)
    rng.shuffle(order)
    return order


def learning_rate_scale(step, max_steps, warmup_ratio=0.0, schedule="constant"):
    warmup = max(1, round(warmup_ratio * max_steps)) if warmup_ratio > 0 else 0
    if step <= warmup:
        return step / warmup
    if schedule == "constant":
        return 1.0
    # First post-warmup step runs at the full rate; the last step stays above zero.
    progress = (step - warmup - 1) / max(1, max_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def fit(model, tokenizer, train, dev, output, *, max_steps=20, grad_accum=4,
        learning_rate=1e-4, max_length=4096, seed=119, eval_every=10, objective=None,
        head_learning_rate=None, warmup_ratio=0.0, schedule="constant", oversample=None,
        select_by="nll", max_downward_rate=None, keep_checkpoints=False, calls_per_batch=1):
    """Bounded single-device supervised fit; choose checkpoints on dev only, never test."""
    import shutil
    import torch
    from .model import reload_adaptation, save_adaptation, training_loss
    rng = random.Random(seed)
    torch.manual_seed(seed)
    parameters = [p for p in model.parameters() if p.requires_grad]
    head_ids = {id(p) for p in model.head.parameters()}
    groups = [{"params": [p for p in parameters if id(p) in head_ids],
               "lr": head_learning_rate or learning_rate},
              {"params": [p for p in parameters if id(p) not in head_ids], "lr": learning_rate}]
    groups = [group for group in groups if group["params"]]
    for group in groups:
        group["initial_lr"] = group["lr"]
    optimizer = torch.optim.AdamW(groups)
    base = evaluate(model, dev, tokenizer, max_length, calls_per_batch=calls_per_batch)
    best_key, best_step = None, None
    gradient_verified = False
    adapter_gradient_verified = False
    history = []
    order, cursor = [], 0
    for step in range(1, max_steps + 1):
        scale = learning_rate_scale(step, max_steps, warmup_ratio, schedule)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * scale
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(grad_accum):
            rows = []
            for _ in range(calls_per_batch):  # several short calls share one padded forward pass
                if cursor == len(order):
                    order, cursor = training_order(train, oversample or {}, rng), 0
                rows.extend(encode(train[order[cursor]], tokenizer, max_length, rng))
                cursor += 1
            loss = training_loss(model(rows), rows, objective)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            (loss / grad_accum).backward()
            total_loss += float(loss.detach()) / grad_accum
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        if any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.head.parameters()):
            gradient_verified = True
        if any(p.grad is not None and bool(p.grad.abs().sum() > 0)
               for name, p in model.backbone.named_parameters() if "lora_" in name):
            adapter_gradient_verified = True
        optimizer.step()
        event = {"step": step, "train_loss": total_loss, "gradient_norm": float(norm), "lr_scale": scale}
        if step % eval_every == 0 or step == max_steps:
            metrics = evaluate(model, dev, tokenizer, max_length, calls_per_batch=calls_per_batch)
            event["dev"] = metrics
            key = selection_key(metrics, select_by, max_downward_rate)
            if keep_checkpoints:
                save_adaptation(model, (output / "checkpoints" / f"step-{step}").resolve())
            if best_key is None or key > best_key:
                best_key, best_step = key, step
                if (output / "adapter").exists():
                    shutil.rmtree(output / "adapter")
                save_adaptation(model, output)
        history.append(event)
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")
        print(json.dumps({"step": step, "train_loss": total_loss}), flush=True)
    if not gradient_verified:
        raise RuntimeError("No nonzero head gradient was observed")
    if any(p.grad is not None for p in model.backbone.parameters() if not p.requires_grad):
        raise RuntimeError("Frozen backbone unexpectedly received gradients")
    if model.train_backbone and not adapter_gradient_verified:
        raise RuntimeError("No nonzero LoRA gradient was observed")
    reload_adaptation(model, output)
    adapted = evaluate(model, dev, tokenizer, max_length, calls_per_batch=calls_per_batch)
    recorded = next(event["dev"] for event in history if event["step"] == best_step)
    if not math.isclose(adapted["mean_nll"], recorded["mean_nll"], rel_tol=1e-5, abs_tol=1e-5):
        raise RuntimeError("Saved/reloaded head failed the dev loss consistency check")
    return {"base_raw_dev": base, "adapted_raw_dev": adapted, "best_step": best_step,
            "improved_dev_nll": adapted["mean_nll"] < base["mean_nll"],
            "nonzero_head_gradient": gradient_verified, "reload_verified": True,
            "nonzero_adapter_gradient": adapter_gradient_verified,
            "base_weights_frozen": True, "steps": max_steps,
            "selection": {"rule": select_by, "max_downward_rate": max_downward_rate,
                          "feasible": best_key[0], "score": best_key[1]}}


def resolve_prompt(variant, guideline_dir, prior=None):
    """A checkpoint is always evaluated with its recorded prompt; renders need the external guideline dir."""
    if prior is not None:
        recorded = prior.get("prompt", {"variant": "bare", "overrides": {}})
        if variant is not None and variant != recorded["variant"]:
            raise ValueError(f"Checkpoint was trained with prompt {recorded['variant']}, not {variant}")
        return recorded
    return render_prompt(variant or "bare", guideline_dir)


def parse_multipliers(text):
    """'other=3,rescue=3' -> {'other': 3, 'rescue': 3}; dispatch labels only."""
    from .prepare import DISPATCH
    result = {}
    for item in filter(None, (text or "").split(",")):
        key, _, value = item.partition("=")
        if key not in DISPATCH.values() or not value.isdigit() or int(value) < 1:
            raise ValueError(f"Invalid oversample entry {item!r}; use dispatch_label=positive_int")
        result[key] = int(value)
    return result


def parse_task_weights(items):
    """['ems_subtype=1.0', ...] -> {'ems_subtype': 1.0}; explicit weights override the defaults."""
    out = {}
    for item in items or []:
        qid, _, value = item.partition("=")
        try:
            weight = float(value)
        except ValueError:
            raise ValueError(f"Invalid --task-weight {item!r}; use question_id=weight") from None
        if not qid or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid --task-weight {item!r}; use question_id=weight with weight >= 0")
        out[qid] = weight
    return out


def checkpoint_questions(requested=None, prior=None):
    """Question IDs to keep: explicit request, else what the checkpoint was trained on, else all (None)."""
    chosen = requested or (prior or {}).get("config", {}).get("questions")
    return {q.strip() for q in chosen.split(",") if q.strip()} if chosen else None


def select_questions(example, keep):
    """Drop questions outside ``keep`` (None keeps all); labels and metadata are untouched."""
    if keep is None:
        return example
    return {**example, "questions": {q: v for q, v in example["questions"].items() if q in keep}}


def build_objective(args, train):
    objective = {"task_weights": {"dispatch": 1.0, "urgency": args.urgency_weight}}
    for qid in sorted({q for e in train for q in e["questions"]}):
        if qid not in PRIMARY:
            objective["task_weights"][qid] = args.aux_weight
    objective["task_weights"].update(parse_task_weights(args.task_weight))
    if args.urgency_label_smoothing:
        objective["label_smoothing"] = {"urgency": args.urgency_label_smoothing}
    if args.urgency_cost_weight:
        objective["cost"] = {"urgency": {"weight": args.urgency_cost_weight, "downward": args.downward_cost}}
    if args.dispatch_logit_adjust:
        counts = Counter(e["questions"]["dispatch"]["label"] for e in train)
        keys = train[0]["questions"]["dispatch"]["criteria"]
        total = sum(counts.values())
        # Unseen classes get a half-count prior so the adjustment stays finite.
        objective["logit_adjust"] = {"dispatch": {
            k: args.dispatch_logit_adjust * math.log(max(counts[k], 0.5) / total) for k in keys}}
    return objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dir", type=Path, help="Explicit local model override; recorded as unverified revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=119)
    parser.add_argument("--mode", choices=["head-only", "lora"], default="head-only")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--evaluate-only", choices=["dev", "confirm", "test"])
    parser.add_argument("--reference-data", type=Path,
                        help="Original checkpoint dataset when evaluating a separately reserved confirm/test split")
    parser.add_argument("--checkpoint", type=Path, help="Adaptation output directory, for evaluate-only")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Completed run to continue training from (its head and LoRA adapter); fresh optimizer")
    parser.add_argument("--lora-targets", choices=("attention-qv", "attention", "attention-mlp"), default="attention-qv")
    parser.add_argument("--head-learning-rate", type=float, help="Defaults to --learning-rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--prompt-variant", choices=VARIANTS,
                        help="Choice text variant; default bare, or the checkpoint's recorded prompt")
    parser.add_argument("--guideline-dir", type=Path,
                        help="External knowledge root containing extracted/chapters (not stored in this repo)")
    parser.add_argument("--urgency-weight", type=float, default=1.0)
    parser.add_argument("--aux-weight", type=float, default=0.5,
                        help="Loss weight of auxiliary questions (e.g. subtype, triage) present in the data")
    parser.add_argument("--urgency-label-smoothing", type=float, default=0.0)
    parser.add_argument("--urgency-cost-weight", type=float, default=0.0,
                        help="Weight of the expected asymmetric ordinal cost term")
    parser.add_argument("--downward-cost", type=float, default=2.0,
                        help="Cost per level of an under-triage error; over-triage costs 1 per level")
    parser.add_argument("--dispatch-logit-adjust", type=float, default=0.0,
                        help="tau for train-time logit adjustment by the dispatch label prior")
    parser.add_argument("--oversample", default="", help="Dispatch label repeats, e.g. other=3,rescue=3")
    parser.add_argument("--select-by", choices=["nll", "composite"], default="nll")
    parser.add_argument("--max-downward-rate", type=float,
                        help="With --select-by composite, prefer dev steps at or below this downward error rate")
    parser.add_argument("--keep-checkpoints", action="store_true", help="Also save every evaluated step")
    parser.add_argument("--task-weight", action="append", default=[],
                        help="Per-question loss weight, repeatable: --task-weight ems_subtype=1.0")
    parser.add_argument("--questions", help="Comma-separated question IDs to keep (default: all in the data)")
    parser.add_argument("--calls-per-batch", type=int, default=1,
                        help="Calls encoded into one padded forward pass (effective batch = this x --grad-accum)")
    args = parser.parse_args()
    if min(args.max_length, args.max_steps, args.grad_accum, args.eval_every, args.lora_rank, args.lora_alpha) <= 0:
        parser.error("Lengths, steps, accumulation and evaluation interval must be positive")
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.urgency_label_smoothing < 1:
        parser.error("Warmup ratio and label smoothing must be in [0, 1)")
    if (min(args.urgency_weight, args.downward_cost) <= 0 or args.urgency_cost_weight < 0
            or args.dispatch_logit_adjust < 0 or args.aux_weight < 0 or args.calls_per_batch < 1):
        parser.error("Loss weights and costs must be nonnegative (task weight and downward cost positive)")
    if args.head_learning_rate is not None and (not math.isfinite(args.head_learning_rate)
                                                or args.head_learning_rate <= 0):
        parser.error("Head learning rate must be finite and positive")
    try:
        oversample = parse_multipliers(args.oversample)
        parse_task_weights(args.task_weight)
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_length > MAX_BRANCH:
        parser.error(f"max_length must be at most {MAX_BRANCH} for the pinned decision protocol")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive")
    if args.checkpoint and not args.evaluate_only:
        parser.error("--checkpoint is for evaluation; use --init-checkpoint to continue training")
    if args.init_checkpoint and (args.evaluate_only or args.checkpoint):
        parser.error("--init-checkpoint is for training runs only")
    if (args.select_by == "composite" and args.questions
            and not set(PRIMARY) <= {q.strip() for q in args.questions.split(",")}):
        parser.error("--select-by composite needs both dispatch and urgency in --questions")
    if args.init_checkpoint:
        init_config = json.loads((args.init_checkpoint / "run.json").read_text(encoding="utf-8")).get("config", {})
        for name in ("lora_rank", "lora_alpha", "lora_targets"):
            if getattr(args, name) != parser.get_default(name) and getattr(args, name) != init_config.get(name):
                parser.error(f"--{name.replace('_', '-')} differs from --init-checkpoint; the loaded adapter keeps its own")
            setattr(args, name, init_config.get(name, getattr(args, name)))
    if args.reference_data and (not args.checkpoint or args.evaluate_only not in ("confirm", "test")):
        parser.error("--reference-data requires a checkpoint and a reserved confirm/test evaluation")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty; use a fresh run directory")
    from .model import enable_lora, init_from_checkpoint, load_zoro, same_base
    import torch
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata = {"mode": args.mode, "status": "started", "config": config,
                "upstream_revision": UPSTREAM_REVISION, "temperature": 1.0, "calibrated_on_119": False,
                "versions": {name: importlib.metadata.version(name)
                             for name in ("torch", "transformers", "huggingface-hub", "peft")}}
    if args.evaluate_only and not args.checkpoint:
        metadata["mode"] = "base"
    write_json(args.output / "run.json", metadata)
    start = time.monotonic()
    try:
        prior = None
        if args.checkpoint:
            prior = json.loads((args.checkpoint / "run.json").read_text(encoding="utf-8"))
            if prior.get("status") != "completed":
                raise ValueError("Checkpoint did not complete successfully")
            if args.max_length != prior["config"]["max_length"]:
                raise ValueError("Use the checkpoint's max_length for evaluation")
            metadata["mode"] = prior["mode"]
        prompt = resolve_prompt(args.prompt_variant, args.guideline_dir, prior)
        # A checkpoint is evaluated on the questions it was trained with unless --questions overrides it.
        keep = checkpoint_questions(args.questions, prior)
        metadata["prompt"] = prompt
        if args.evaluate_only:
            examples, manifest = evaluation_split(args.data, args.evaluate_only, prior, args.reference_data)
            examples = [select_questions(apply_prompt(e, prompt), keep) for e in examples]
        else:
            train, manifest = load_split(args.data, "train")
            dev, _ = load_split(args.data, "dev")
            for key in ("group_id", "record_id", "content_hash"):
                if {r["_meta"][key] for r in train} & {r["_meta"][key] for r in dev}:
                    raise ValueError(f"Train/dev overlap in {key}")
            train = [select_questions(apply_prompt(e, prompt), keep) for e in train]
            dev = [select_questions(apply_prompt(e, prompt), keep) for e in dev]
        metadata["dataset_manifest"] = manifest
        model, tokenizer, identity = load_zoro(device=args.device, model_dir=args.model_dir,
            head_path=args.checkpoint / "decision_head.pt" if args.checkpoint else None,
            adapter_path=args.checkpoint / "adapter" if prior and prior["mode"] == "lora" else None,
            local_files_only=args.local_files_only)
        metadata["model"] = identity
        if prior and not same_base(prior["model"], identity):
            raise ValueError("Checkpoint uses a different base model identity")
        if args.init_checkpoint:
            init = json.loads((args.init_checkpoint / "run.json").read_text(encoding="utf-8"))
            if init.get("status") != "completed" or init["mode"] != args.mode:
                raise ValueError("--init-checkpoint must be a completed run of the same --mode")
            if not same_base(init["model"], identity):
                raise ValueError("--init-checkpoint uses a different base model identity")
            if init.get("prompt") != prompt:
                raise ValueError("--init-checkpoint was trained with another prompt; pass the same --prompt-variant")
            if checkpoint_questions(None, init) != keep:
                raise ValueError("--init-checkpoint was trained on another question set; pass the same --questions")
            if args.mode == "lora" and not (args.init_checkpoint / "adapter").is_dir():
                raise ValueError("--init-checkpoint has no LoRA adapter to continue")
            files = [args.init_checkpoint / "decision_head.pt"] + (
                [args.init_checkpoint / "adapter" / "adapter_model.safetensors"] if args.mode == "lora" else [])
            metadata["init_checkpoint"] = {"path": str(args.init_checkpoint), "best_step": init["result"].get("best_step"),
                                           "dataset_train_sha256": init["dataset_manifest"]["splits"]["train"]["sha256"],
                                           "sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
            init_from_checkpoint(model, args.init_checkpoint)
        elif args.mode == "lora" and not args.evaluate_only:
            enable_lora(model, args.lora_rank, args.lora_alpha, args.lora_targets)
        metadata["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        metadata["trainable_names"] = [name for name, p in model.named_parameters() if p.requires_grad]
        if args.evaluate_only:
            examples, coverage = filter_examples(examples, tokenizer, args.max_length)
            result = {"split": args.evaluate_only, "coverage": coverage,
                      "metrics": evaluate(model, examples, tokenizer, args.max_length,
                                          calls_per_batch=args.calls_per_batch)}
        else:
            train, train_coverage = filter_examples(train, tokenizer, args.max_length)
            dev, dev_coverage = filter_examples(dev, tokenizer, args.max_length)
            objective = build_objective(args, train)
            metadata["objective"] = objective
            write_json(args.output / "run.json", metadata)
            result = fit(model, tokenizer, train, dev, args.output, max_steps=args.max_steps,
                         grad_accum=args.grad_accum, learning_rate=args.learning_rate,
                         max_length=args.max_length, seed=args.seed, eval_every=args.eval_every,
                         objective=objective, head_learning_rate=args.head_learning_rate,
                         warmup_ratio=args.warmup_ratio, schedule=args.schedule, oversample=oversample,
                         select_by=args.select_by, max_downward_rate=args.max_downward_rate,
                         keep_checkpoints=args.keep_checkpoints, calls_per_batch=args.calls_per_batch)
            result["coverage"] = {"train": train_coverage, "dev": dev_coverage}
            write_json(args.output / "temperature.json", {"text": 1.0})
        metadata.update(status="completed", elapsed_seconds=time.monotonic() - start,
                        dataset_manifest=manifest, result=result)
        write_json(args.output / "run.json", metadata)
        print(json.dumps({"status": "completed", "output": str(args.output)}))
    except Exception as exc:
        metadata.update(status="failed", elapsed_seconds=time.monotonic() - start, error_type=type(exc).__name__)
        write_json(args.output / "run.json", metadata)
        raise


if __name__ == "__main__":
    main()
