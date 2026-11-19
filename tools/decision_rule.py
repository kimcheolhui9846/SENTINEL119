"""Tune per-class log-score offsets on exported decision scores (CPU, no model needed).

Training changes the scores; this changes only the decision made from them. Offsets are added to
log-softmax scores before argmax. Urgency maximizes macro-F1 subject to a downward (under-triage)
error-rate cap; dispatch maximizes macro-F1. Performance on the tuning split is estimated with
k-fold cross-fitting (offsets chosen on k-1 folds, applied to the held-out fold), because offsets
chosen and scored on the same calls are optimistic. A final policy fitted on all calls is saved
for one later application to a held-out split (``--apply``).

  python tools/decision_rule.py --data D --split confirm --scores S.json --output policy.json [--max-downward 0.2]
  python tools/decision_rule.py --data D --split test --scores T.json --apply policy.json --output test.json
"""

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import random

import numpy as np

URGENCY = ["high", "medium", "low"]
DISPATCH = ["fire", "rescue", "ems", "other"]
RANK = {"high": 0, "medium": 1, "low": 2}


def log_softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def metrics(truth, pred, labels):
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for a, p in zip(truth, pred):
        m[a, p] += 1
    f1 = [2 * m[i, i] / (m[i].sum() + m[:, i].sum()) if m[i].sum() + m[:, i].sum() else 0.0 for i in range(len(labels))]
    out = {"accuracy": float(np.trace(m) / len(truth)), "macro_f1": float(np.mean(f1)),
           "confusion_matrix": m.tolist(), "labels": labels}
    if labels == URGENCY:
        out["downward"] = int(sum(m[a, p] for a in range(3) for p in range(3) if p > a))
        out["high_to_low"] = int(m[0, 2])
        out["downward_rate"] = out["downward"] / len(truth)
    return out


def identity(scores):
    """What a policy was fitted on: base model, checkpoint files, prompt and question set."""
    model = scores.get("model") or {}
    return {"model": {k: model.get(k) for k in ("model_id", "revision", "config_sha256", "base_head_sha256")},
            "checkpoint_sha256": scores.get("checkpoint_sha256") or {}, "prompt": (scores.get("prompt") or {}).get("variant"),
            "questions": scores.get("questions")}


def load(data, split, scores_path):
    payload = (data / f"{split}.jsonl").read_bytes()
    rows = {}
    for line in payload.decode("utf-8").split("\n"):  # not splitlines(): U+2028 may occur inside strings
        if line.strip():
            e = json.loads(line)
            rows[e["_meta"]["group_id"]] = {q: v["label"] for q, v in e["questions"].items()}
    scores = json.loads(Path(scores_path).read_text(encoding="utf-8"))
    if scores.get("split") != split:
        raise ValueError(f"Scores are for split {scores.get('split')!r}, not {split!r}")
    if scores.get("data_sha256") != hashlib.sha256(payload).hexdigest():
        raise ValueError("Scores were exported from a different version of this split")
    ids = [s["group_id"] for s in scores["scores"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Scores contain duplicate calls")
    if set(ids) != set(rows):
        raise ValueError(f"Scores and split differ: {len(set(rows) - set(ids))} missing, {len(set(ids) - set(rows))} extra")
    out = {"ids": ids, "identity": identity(scores)}
    for qid, labels in (("urgency", URGENCY), ("dispatch", DISPATCH)):
        out[qid] = (log_softmax(np.array([[s["logits"][qid][k] for k in labels] for s in scores["scores"]])),
                    np.array([labels.index(rows[i][qid]) for i in ids]))
    if all("subtype" in s["logits"] for s in scores["scores"]):
        from sentinel119.auxiliary import SUBTYPE_DISPATCH
        derived = []
        for s in scores["scores"]:
            sub = s["logits"]["subtype"]
            keys = list(sub)
            p = np.exp(log_softmax(np.array([[sub[k] for k in keys]])))[0]
            derived.append([np.log(sum(p[j] for j, k in enumerate(keys) if SUBTYPE_DISPATCH[k] == d) + 1e-9)
                            for d in DISPATCH])
        out["dispatch_from_subtype"] = (np.array(derived), out["dispatch"][1])
    return out


def decide(scores, offsets):
    return (scores + np.asarray(offsets)).argmax(axis=1)


def grid(task):
    if task == "urgency":  # offset for medium fixed at 0
        steps = np.round(np.arange(-1.5, 1.51, 0.05), 3)
        return [(h, 0.0, l) for h, l in product(steps, steps)]
    steps = np.round(np.arange(-1.0, 5.01, 0.25), 3)  # ems fixed at 0; boost rarer classes
    return [(f, r, 0.0, o) for f, r, o in product(steps, steps, steps)]


def fit(task, scores, truth, max_downward=None):
    labels = URGENCY if task == "urgency" else DISPATCH
    best = None
    for offsets in grid(task):
        m = metrics(truth, decide(scores, offsets), labels)
        feasible = max_downward is None or m["downward_rate"] <= max_downward
        key = (feasible, m["macro_f1"], -sum(abs(o) for o in offsets))
        if best is None or key > best[0]:
            best = (key, offsets)
    return [float(o) for o in best[1]], bool(best[0][0])


def crossfit(task, scores, truth, max_downward, k=5, seed=0):
    order = list(range(len(truth)))
    random.Random(seed).shuffle(order)
    pred = np.zeros(len(truth), dtype=int)
    for f in range(k):
        held = order[f::k]
        train = np.setdiff1d(np.arange(len(truth)), held)
        offsets, _ = fit(task, scores[train], truth[train], max_downward)
        pred[held] = decide(scores[held], offsets)
    return pred


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-downward", type=float, help="Urgency downward-error-rate cap during tuning")
    parser.add_argument("--apply", type=Path, help="Apply a saved policy instead of tuning")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output file")
    if args.split == "test" and not args.apply:
        parser.error("Tuning on the final test split is not allowed; tune on dev/confirm and pass --apply")
    data = load(args.data, args.split, args.scores)
    policy = None
    if args.apply:
        policy = json.loads(args.apply.read_text(encoding="utf-8"))
        if policy.get("identity") != data["identity"]:
            parser.error("Policy was fitted on scores from a different model, checkpoint, prompt or question set")
    tasks = [t for t in ("urgency", "dispatch", "dispatch_from_subtype") if t in data]
    report = {"split": args.split, "calls": len(data["ids"]), "scores": str(args.scores), "identity": data["identity"],
              "raw": {}, "tasks": {}}
    for task in tasks:
        scores, truth = data[task]
        labels = URGENCY if task == "urgency" else DISPATCH
        report["raw"][task] = metrics(truth, decide(scores, np.zeros(len(labels))), labels)
        if args.apply:
            offsets = policy["tasks"][task]["offsets"]
            report["tasks"][task] = {"offsets": offsets, "applied": metrics(truth, decide(scores, offsets), labels)}
            continue
        cap = args.max_downward if task == "urgency" else None
        offsets, feasible = fit(task, scores, truth, cap)
        report["tasks"][task] = {"offsets": offsets, "cap_feasible_on_all": feasible, "max_downward": cap,
                                 "crossfit_estimate": metrics(truth, crossfit(task, scores, truth, cap), labels),
                                 "in_sample_optimistic": metrics(truth, decide(scores, offsets), labels)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for task in tasks:
        raw, t = report["raw"][task], report["tasks"][task]
        tuned = t.get("applied") or t["crossfit_estimate"]
        extra = lambda m: f" down={m['downward']}" if "downward" in m else ""
        print(f"{task:22s} raw F1={raw['macro_f1']:.4f}{extra(raw)}  ->  "
              f"{'applied' if args.apply else 'crossfit'} F1={tuned['macro_f1']:.4f}{extra(tuned)}  offsets={t['offsets']}")


if __name__ == "__main__":
    main()
