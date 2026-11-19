"""Hierarchical full-call targets: dispatch -> subtype (EMS, rescue) -> urgency, plus an EMS main-symptom group.

Product order: decide the dispatch type first; only for 구급 / 구조 ask the subtype among that type's children
(AI Hub ``disasterMedium``). Fire subtypes carry almost no urgency or dispatch information (일반화재 86%, all
≥93% 상), so fire and 기타 stop at the dispatch type. For EMS calls a single main-symptom group is derived from
the AI Hub ``symptom`` list by severity priority (``none`` when the list is empty).

Metadata are training targets only; the model input stays the time-ordered transcript. Evaluation splits reuse
the exact calls of an existing prepared dataset (same group IDs and order) so results remain comparable.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import zipfile

from .prepare import DISPATCH, digest, eligible_groups, make_example, scan
from .train import write_json

EMS_SUBTYPES = {"질병(중증)": "severe_illness", "질병(중증 외)": "illness", "심정지": "cardiac_arrest",
                "부상": "injury", "사고": "accident", "임산부": "pregnancy", "약물중독": "poisoning",
                "기타구급": "other_ems"}
RESCUE_SUBTYPES = {"안전사고": "safety", "대물사고": "property", "자살": "suicide", "기타구조": "other_rescue"}
# Highest-priority group present wins. Order follows dispatch-stage severity, not frequency.
SYMPTOM_GROUPS = [
    ("arrest", "심정지·호흡정지", {"심정지", "호흡정지"}),
    ("consciousness", "의식저하·실신", {"의식장애", "실신"}),
    ("seizure", "경련·발작", {"경련/발작"}),
    ("breathing", "호흡곤란·기도이물", {"호흡곤란", "기도이물"}),
    ("chest", "흉통·가슴불편감·심계항진", {"흉통", "가슴불편감", "심계항진"}),
    ("paralysis", "마비", {"마비"}),
    ("bleeding", "출혈(토혈·혈변·객혈·코피 등)", {"그 밖의출혈", "비출혈", "토혈", "혈변", "객혈"}),
    ("obstetric", "분만진통·질출혈", {"분만진통", "질출혈"}),
    ("trauma", "외상(열상·골절·화상 등)", {"열상", "찰과상", "타박상", "골절", "탈구", "염좌", "절단", "압궤손상", "화상", "경추통증"}),
    ("fever", "고열·기침", {"고열", "기침"}),
    ("digestive", "복통·구토·설사", {"복통", "구토", "오심", "설사", "변비"}),
    ("dizziness", "어지러움·전신쇠약·두통", {"어지러움", "전신쇠약", "두통"}),
    ("pain", "기타 통증", {"기타통증", "요통", "그밖의통증기타"}),
    ("other", "기타 증상", None),  # any remaining symptom
    ("none", "기록된 증상 없음", set()),
]
QUESTIONS = {
    "ems_subtype": {"type": "choice", "instructions": "구급 신고입니다. 이 대화의 구급 세부유형을 AI Hub 분류 기준으로 고르세요.",
                    "criteria": {key: name for name, key in EMS_SUBTYPES.items()}},
    "rescue_subtype": {"type": "choice", "instructions": "구조 신고입니다. 이 대화의 구조 세부유형을 AI Hub 분류 기준으로 고르세요.",
                       "criteria": {key: name for name, key in RESCUE_SUBTYPES.items()}},
    "ems_symptom": {"type": "choice", "instructions": "구급 신고입니다. 이 대화에서 확인되는 가장 위중한 주증상 그룹을 고르세요.",
                    "criteria": {key: name for key, name, _ in SYMPTOM_GROUPS}},
}
CHILD_QUESTIONS = {"ems": ("ems_subtype", "ems_symptom"), "rescue": ("rescue_subtype",)}


def symptom_group(symptoms):
    present = {s.strip() for s in symptoms or [] if isinstance(s, str) and s.strip()}
    if not present:
        return "none"
    for key, _, members in SYMPTOM_GROUPS:
        if members and present & members:
            return key
    return "other"


def targets(raw):
    """Hierarchical extra labels for one raw record, keyed by question id."""
    dispatch = DISPATCH.get(raw.get("disasterLarge"))
    medium = raw.get("disasterMedium")
    out = {}
    if dispatch == "ems":
        if medium not in EMS_SUBTYPES:
            raise ValueError(f"Unknown EMS subtype {medium!r}")
        out["ems_subtype"] = EMS_SUBTYPES[medium]
        out["ems_symptom"] = symptom_group(raw.get("symptom"))
    elif dispatch == "rescue":
        if medium not in RESCUE_SUBTYPES:
            raise ValueError(f"Unknown rescue subtype {medium!r}")
        out["rescue_subtype"] = RESCUE_SUBTYPES[medium]
    return out


def hierarchical_example(raw, group_id, source):
    example = make_example(raw, group_id, source)
    for qid, label in targets(raw).items():
        example["questions"][qid] = {**QUESTIONS[qid], "label": label}
    return example


def read_raw(archives, items, extract=None):
    """items: [(archive name or Path, member)] -> {(archive name, member): extract(raw)}, one open per ZIP."""
    by_archive = defaultdict(list)
    for archive, member in items:
        by_archive[Path(archive).name].append(member)
    out = {}
    for name, members in by_archive.items():
        with zipfile.ZipFile(archives[name]) as z:
            for member in members:
                raw = json.loads(z.read(member).decode("utf-8-sig"))
                out[(name, member)] = extract(raw) if extract else raw
    return out


def label_fields(raw):
    return (raw.get("disasterLarge"), raw.get("disasterMedium"), raw.get("urgencyLevel"))


def allocate(pool_keys, count, floors, seed):
    """Proportional allocation over strata with per-stratum floors (capped by availability)."""
    sizes = Counter(pool_keys)
    quota = {k: min(sizes[k], floors.get(k, 0)) for k in sizes}
    remaining = count - sum(quota.values())
    if remaining < 0:
        raise ValueError("Floors exceed the requested count")
    free = {k: sizes[k] - quota[k] for k in sizes}
    total_free = sum(free.values())
    shares = {k: remaining * free[k] / total_free for k in sizes} if total_free else {}
    for k, share in shares.items():
        quota[k] += min(free[k], int(share))
    leftover = count - sum(quota.values())
    for k in sorted(shares, key=lambda k: (-(shares[k] - int(shares[k])), str(k))):
        if leftover <= 0:
            break
        if quota[k] < sizes[k]:
            quota[k] += 1
            leftover -= 1
    return quota


def write_split(output, name, examples):
    path = output / f"{name}.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    labels = {qid: dict(Counter(e["questions"][qid]["label"] for e in examples if qid in e["questions"]))
              for qid in ("dispatch", "urgency", *QUESTIONS)}
    return {"calls": len(examples), "labels": labels, "missing_labels": {},
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare_hierarchy(data_root, eval_data, output, train_count=8000, seed=219, floors=None):
    """New stratified train set + hierarchical questions on ``eval_data``'s dev/confirm/test calls."""
    if output.exists():
        raise ValueError("Use a new output directory")
    eval_manifest = json.loads((eval_data / "manifest.json").read_text(encoding="utf-8"))
    archives = {p.name: p for p in data_root.rglob("*.zip")}
    reserved_rows = {}
    for name in ("dev", "confirm", "test"):
        payload = (eval_data / f"{name}.jsonl").read_bytes()
        if hashlib.sha256(payload).hexdigest() != eval_manifest["splits"][name]["sha256"]:
            raise ValueError(f"{name}.jsonl does not match its manifest")
        reserved_rows[name] = [json.loads(line) for line in payload.decode("utf-8").split("\n") if line.strip()]
    reserved = {k: {r["_meta"][k] for rows in reserved_rows.values() for r in rows}
                for k in ("group_id", "record_id", "content_hash")}

    entries, groups, report = scan(data_root)
    pools, audit = eligible_groups(entries, groups)
    candidates = [(gid, e) for gid, e in pools["Training"]
                  if gid not in reserved["group_id"] and e.record_id not in reserved["record_id"]
                  and e.content_hash not in reserved["content_hash"]]
    # Only label fields for all candidates; full records are re-read for the chosen calls.
    fields = read_raw(archives, [(e.archive, e.member) for _, e in candidates], label_fields)
    keyed = [(fields[(Path(e.archive).name, e.member)], gid, e) for gid, e in candidates]
    # Floors are per (dispatch, subtype); spread over urgency in proportion within the subtype.
    floors = floors if floors is not None else {("구급", "임산부"): 240, ("구급", "약물중독"): 203, ("구조", "자살"): 320,
                        ("구급", "사고"): 500, ("구급", "심정지"): 500, ("기타", "기타"): 600,
                        ("구조", "대물사고"): 500}
    by_sub = Counter(k[:2] for k, *_ in keyed)
    stratum_floor = {}
    for k, n in Counter(k for k, *_ in keyed).items():
        if k[:2] in floors:
            stratum_floor[k] = round(floors[k[:2]] * n / by_sub[k[:2]])
    quota = allocate([k for k, *_ in keyed], train_count, stratum_floor, seed)
    # Deterministic within-stratum order from a seeded hash of the group ID.
    taken, chosen = Counter(), []
    for k, gid, e in sorted(keyed, key=lambda item: digest(f"{seed}:{item[1]}")):
        if taken[k] < quota.get(k, 0):
            taken[k] += 1
            chosen.append((gid, e))
    random.Random(seed).shuffle(chosen)
    raws = read_raw(archives, [(e.archive, e.member) for _, e in chosen])
    train = [hierarchical_example(raws[(Path(e.archive).name, e.member)], gid,
                                  {"archive": Path(e.archive).name, "member": e.member, "official_split": e.official})
             for gid, e in chosen]
    if len(train) != train_count:
        raise ValueError(f"Selected {len(train)} calls, expected {train_count}")

    eval_examples = {}
    for name, rows in reserved_rows.items():
        raws_eval = read_raw(archives, [(r["_meta"]["source"]["archive"], r["_meta"]["source"]["member"]) for r in rows])
        out = []
        for r in rows:
            raw = raws_eval[(r["_meta"]["source"]["archive"], r["_meta"]["source"]["member"])]
            if raw.get("recordId") != r["_meta"]["record_id"]:
                raise ValueError("Source record changed since preparation")
            example = hierarchical_example(raw, r["_meta"]["group_id"], r["_meta"]["source"])
            if example["state"] != r["state"] or any(example["questions"][q]["label"] != r["questions"][q]["label"]
                                                     for q in ("dispatch", "urgency")):
                raise ValueError("Re-derived evaluation example differs from the prepared one")
            out.append(example)
        eval_examples[name] = out

    output.mkdir(parents=True)
    splits = {"train": write_split(output, "train", train)}
    for name, rows in eval_examples.items():
        splits[name] = write_split(output, name, rows)
    report.update(audit)
    report.update({"schema_version": 1, "seed": seed, "label_scope": "full_call", "hierarchy": True,
                   "train_floors": {f"{a}/{b}": n for (a, b), n in floors.items()},
                   "train_subtypes": {f"{a}/{b}": n for (a, b), n in Counter(k[:2] for k in taken.elements()).items()},
                   "evaluation_source": {"path": eval_data.name, "splits": {k: eval_manifest["splits"][k] for k in reserved_rows}},
                   "excluded_reserved_groups": len(reserved["group_id"]), "splits": splits})
    write_json(output / "manifest.json", report)
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True, help="Prepared dataset with dev/confirm/test to reuse")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=219)
    args = parser.parse_args()
    splits = prepare_hierarchy(args.data_root, args.eval_data, args.output, args.train_count, args.seed)
    print(json.dumps({k: v["calls"] for k, v in splits.items()}))


if __name__ == "__main__":
    main()
