"""Read AI Hub 71768 label ZIPs without extracting or modifying the source data."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import unicodedata
import zipfile

DISPATCH = {"화재": "fire", "구조": "rescue", "구급": "ems", "기타": "other"}
URGENCY = {"상": "high", "중": "medium", "하": "low"}
QUESTIONS = {
    "dispatch": {"type": "choice", "instructions": "이 119 신고 대화의 출동유형을 분류하세요.",
                 "criteria": {v: k for k, v in DISPATCH.items()}},
    "urgency": {"type": "choice", "instructions": "이 119 신고 대화의 긴급도를 AI Hub 분류 기준의 상·중·하로 분류하세요.",
                "criteria": {v: k for k, v in URGENCY.items()}},
}


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value):
    return " ".join(unicodedata.normalize("NFKC", value).split())


def normalize_call(raw):
    """Validate labels and retain original turn IDs/times; never use metadata as model input."""
    record_id = raw.get("recordId")
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError("missing_record_id")
    if raw.get("disasterLarge") not in DISPATCH or raw.get("urgencyLevel") not in URGENCY:
        raise ValueError("unknown_label")
    utterances = raw.get("utterances")
    if not isinstance(utterances, list) or not utterances:
        raise ValueError("empty_utterances")
    turns = []
    for index, item in enumerate(utterances):
        if not isinstance(item, dict):
            raise ValueError("invalid_turn")
        text = item.get("text")
        if not isinstance(text, str) or not normalize_text(text):
            raise ValueError("empty_turn")
        start, end = item.get("startAt"), item.get("endAt")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end)):
            raise ValueError("invalid_time")
        if start < 0 or end < start:
            raise ValueError("invalid_time")
        if type(item.get("speaker")) is not int or item["speaker"] not in (0, 1):
            raise ValueError("unknown_speaker")
        turn_id = item.get("id")
        if not isinstance(turn_id, str) or not turn_id:
            raise ValueError("missing_turn_id")
        turns.append({"id": turn_id, "startAt": start, "endAt": end,
                      "speaker": item["speaker"], "text": text, "source_index": index})
    ordered = sorted(turns, key=lambda t: (t["startAt"], t["endAt"], t["source_index"]))
    # Ignore speaker assignments and turn boundaries for duplicate-text grouping.
    content_hash = digest(normalize_text(" ".join(t["text"] for t in ordered)).casefold())
    return {"record_id": record_id, "turns": ordered, "content_hash": content_hash,
            "labels": (DISPATCH[raw["disasterLarge"]], URGENCY[raw["urgencyLevel"]]),
            "reordered": ordered != turns}


@dataclass
class Entry:
    archive: Path
    member: str
    official: str
    record_id: str
    content_hash: str
    labels: tuple


class Groups:
    """Union both record IDs and transcript hashes, including transitive duplicates."""

    def __init__(self):
        self.parent = []
        self.seen = {}

    def find(self, index):
        while index != self.parent[index]:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def add(self, record_id, content_hash):
        index = len(self.parent)
        self.parent.append(index)
        for key in (("id", record_id), ("text", content_hash)):
            if key in self.seen:
                self.parent[self.find(index)] = self.find(self.seen[key])
            self.seen[key] = index


def scan(root):
    archives = sorted(p for p in root.rglob("*.zip") if p.name.startswith(("TL_", "VL_")))
    if not archives:
        raise ValueError("No TL_*.zip / VL_*.zip label archives found")
    entries, groups = [], Groups()
    report = {"archives": [], "rejected": Counter(), "reordered_calls": 0,
              "nonempty_symptom": 0, "nonempty_triage": 0}
    for path in archives:
        official = "Training" if path.name.startswith("TL_") else "Validation"
        with zipfile.ZipFile(path) as archive:
            members = sorted(n for n in archive.namelist() if n.lower().endswith(".json"))
            report["archives"].append({"name": path.name, "official": official,
                                       "json_count": len(members), "bytes": path.stat().st_size})
            for member in members:
                try:
                    raw = json.loads(archive.read(member).decode("utf-8-sig"))
                    if not isinstance(raw, dict):
                        raise ValueError("invalid_document")
                    call = normalize_call(raw)
                except (ValueError, UnicodeError) as exc:
                    reason = str(exc) if str(exc) in {
                        "missing_record_id", "unknown_label", "empty_utterances", "invalid_turn",
                        "empty_turn", "invalid_time", "unknown_speaker", "missing_turn_id", "invalid_document"
                    } else "invalid_json"
                    report["rejected"][reason] += 1
                    continue
                report["reordered_calls"] += call["reordered"]
                report["nonempty_symptom"] += bool(raw.get("symptom"))
                report["nonempty_triage"] += bool(raw.get("triage"))
                entries.append(Entry(path, member, official, call["record_id"], call["content_hash"], call["labels"]))
                groups.add(call["record_id"], call["content_hash"])
    return entries, groups, report


def eligible_groups(entries, groups):
    components = defaultdict(list)
    for index, entry in enumerate(entries):
        components[groups.find(index)].append(entry)
    pools = {"Training": [], "Validation": []}
    counts = Counter()
    for component in components.values():
        if len({e.official for e in component}) > 1:
            counts["cross_official_groups_excluded"] += 1
            counts["cross_official_records_excluded"] += len(component)
            continue
        if len({e.labels for e in component}) > 1:
            counts["conflicting_label_groups_excluded"] += 1
            counts["conflicting_label_records_excluded"] += len(component)
            continue
        representative = min(component, key=lambda e: (e.record_id, e.content_hash, str(e.archive), e.member))
        group_id = digest("\n".join(sorted({"id:" + e.record_id for e in component} |
                                         {"text:" + e.content_hash for e in component})))
        pools[representative.official].append((group_id, representative))
        counts["duplicate_records_collapsed"] += len(component) - 1
    return pools, dict(counts)


def stratified_take(pool, count, seed):
    """Deterministic proportional sampling with one per joint stratum when budget permits."""
    buckets = defaultdict(list)
    for item in sorted(pool, key=lambda x: x[0]):
        buckets[item[1].labels].append(item)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    count = min(count, len(pool))
    quotas = {key: 0 for key in buckets}
    if count >= len(buckets):
        quotas = {key: 1 for key in buckets}
    else:
        # Tiny smoke sets may have fewer slots than joint strata. Cover marginal
        # dispatch/urgency labels first, instead of silently losing rare classes.
        uncovered = {(i, key[i]) for key in buckets for i in (0, 1)}
        while uncovered and sum(quotas.values()) < count:
            key = max(buckets, key=lambda k: (
                sum((i, k[i]) in uncovered for i in (0, 1)), len(buckets[k]), k))
            quotas[key] = 1
            uncovered -= {(i, key[i]) for i in (0, 1)}
    for _ in range(count - sum(quotas.values())):
        key = max((k for k in buckets if quotas[k] < len(buckets[k])),
                  key=lambda k: (count * len(buckets[k]) / len(pool) - quotas[k], k))
        quotas[key] += 1
    selected, remaining = [], []
    for key in sorted(buckets):
        selected.extend(buckets[key][:quotas[key]])
        remaining.extend(buckets[key][quotas[key]:])
    rng.shuffle(selected)
    return selected, remaining


def make_example(raw, group_id, source):
    call = normalize_call(raw)
    questions = {name: {**question, "label": label}
                 for (name, question), label in zip(QUESTIONS.items(), call["labels"])}
    return {"state": "\n".join(f"화자{t['speaker']}: {t['text']}" for t in call["turns"]),
            "questions": questions,
            "_meta": {"group_id": group_id, "record_id": call["record_id"],
                      "content_hash": call["content_hash"], "label_scope": "full_call",
                      "source": source, "turns": call["turns"]}}


def write_split(output, split, selected):
    by_archive = defaultdict(list)
    for group_id, entry in selected:
        by_archive[entry.archive].append((group_id, entry))
    examples = {}
    for path, items in by_archive.items():
        with zipfile.ZipFile(path) as archive:
            for group_id, entry in items:
                raw = json.loads(archive.read(entry.member).decode("utf-8-sig"))
                current = normalize_call(raw)
                if (current["record_id"], current["content_hash"], current["labels"]) != (
                        entry.record_id, entry.content_hash, entry.labels):
                    raise ValueError("Source changed after the split audit; rerun into a fresh directory")
                examples[group_id] = make_example(raw, group_id, {
                    "archive": path.name, "member": entry.member, "official_split": entry.official})
    with (output / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for group_id, _ in selected:
            handle.write(json.dumps(examples[group_id], ensure_ascii=False) + "\n")
    counts = {name: dict(Counter(e.labels[i] for _, e in selected))
              for i, name in enumerate(QUESTIONS)}
    missing = {name: sorted(set(q["criteria"]) - set(counts[name])) for name, q in QUESTIONS.items()}
    return {"calls": len(selected), "labels": counts, "missing_labels": missing,
            "sha256": hashlib.sha256((output / f"{split}.jsonl").read_bytes()).hexdigest()}


def prepare(root, output, *, train_count=80, dev_count=10, test_count=10, seed=119):
    if any(n <= 0 for n in (train_count, dev_count, test_count)):
        raise ValueError("All split counts must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; use a new dataset directory")
    entries, groups, report = scan(root)
    pools, group_report = eligible_groups(entries, groups)
    dev, rest = stratified_take(pools["Training"], dev_count, seed)
    train, _ = stratified_take(rest, train_count, seed + 1)
    test, _ = stratified_take(pools["Validation"], test_count, seed + 2)
    splits = {"train": train, "dev": dev, "test": test}
    if not all(splits.values()):
        raise ValueError("Not enough eligible calls for nonempty train/dev/test splits")
    report.update(group_report)
    report.update({"schema_version": 1, "seed": seed, "label_scope": "full_call",
                   "valid_records": len(entries), "group_counts": {k: len(v) for k, v in pools.items()},
                   "requested": dict(zip(splits, (train_count, dev_count, test_count))), "splits": {}})
    output.mkdir(parents=True, exist_ok=True)
    for split, selected in splits.items():
        report["splits"][split] = write_split(output, split, selected)
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=80)
    parser.add_argument("--dev-count", type=int, default=10)
    parser.add_argument("--test-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=119)
    args = parser.parse_args()
    report = prepare(args.data_root, args.output, train_count=args.train_count,
                     dev_count=args.dev_count, test_count=args.test_count, seed=args.seed)
    print(json.dumps({"output": str(args.output), "splits": {k: v["calls"] for k, v in report["splits"].items()},
                      "rejected": report["rejected"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
