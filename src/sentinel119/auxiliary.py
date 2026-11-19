"""Add auxiliary subtype and triage choice questions to a prepared full-call dataset.

AI Hub 71768 records carry human-annotated ``disasterMedium`` (16 subtypes nested in the four
dispatch types) and ``triage`` (empty for most non-EMS calls). They are used only as extra
training targets, never as model input. Calls, order and split membership are unchanged; only
the question set grows, so results stay comparable with checkpoints trained on the parent data.

Evaluating a dispatch/urgency-only checkpoint on augmented data is unaffected: each question is
encoded as its own causal row, so the dispatch and urgency rows are byte-identical.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

from .prepare import DISPATCH
from .train import write_json

# (disasterLarge, disasterMedium) -> stable key. Order is the option order before shuffling.
SUBTYPES = {
    ("화재", "일반화재"): "fire_general", ("화재", "기타화재"): "fire_other", ("화재", "산불"): "fire_wildfire",
    ("구조", "안전사고"): "rescue_safety", ("구조", "대물사고"): "rescue_property",
    ("구조", "자살"): "rescue_suicide", ("구조", "기타구조"): "rescue_other",
    ("구급", "질병(중증)"): "ems_severe_illness", ("구급", "질병(중증 외)"): "ems_illness",
    ("구급", "심정지"): "ems_cardiac_arrest", ("구급", "부상"): "ems_injury", ("구급", "사고"): "ems_accident",
    ("구급", "임산부"): "ems_pregnancy", ("구급", "약물중독"): "ems_poisoning", ("구급", "기타구급"): "ems_other",
    ("기타", "기타"): "other_other",
}
SUBTYPE_DISPATCH = {key: DISPATCH[large] for (large, _), key in SUBTYPES.items()}
TRIAGE = {"응급증상": "emergent", "준응급증상": "urgent", "잠재응급증상": "less_urgent", "대상외": "not_applicable",
          "사망추정": "presumed_dead", "사망": "dead", "": "unrecorded"}
AUX_QUESTIONS = {
    "subtype": {"type": "choice",
                "instructions": "이 119 신고 대화의 출동 세부유형을 AI Hub 분류 기준으로 고르세요.",
                "criteria": {key: f"{large} {medium}" for (large, medium), key in SUBTYPES.items()}},
    "triage": {"type": "choice",
               "instructions": ("이 119 신고 대화의 응급증상 분류를 AI Hub 기록 기준으로 고르세요. "
                                "기록이 없는 신고는 미기재입니다."),
               "criteria": {key: (name or "미기재") for name, key in TRIAGE.items()}},
}


def labels_for(raw, dispatch_label):
    pair = (raw.get("disasterLarge"), raw.get("disasterMedium"))
    if pair not in SUBTYPES:
        raise ValueError(f"Unknown subtype {pair}")
    subtype = SUBTYPES[pair]
    if SUBTYPE_DISPATCH[subtype] != dispatch_label:
        raise ValueError("Subtype does not nest under the prepared dispatch label")
    triage = (raw.get("triage") or "").strip()
    if triage not in TRIAGE:
        raise ValueError(f"Unknown triage value {triage!r}")
    return subtype, TRIAGE[triage]


def augment(data_root, data, output, parent=None):
    """Write ``output`` = ``data`` + subtype/triage questions; ``parent`` is an already augmented parent.

    Builds in a staging directory and renames it at the end, so a failed run leaves nothing at ``output``."""
    import shutil
    if output.exists():
        raise ValueError("Use a new output directory")
    staging = output.with_name(output.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    try:
        counts = _augment(data_root, data, staging, parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    staging.rename(output)
    return counts


def _augment(data_root, data, output, parent):
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    archives = {path.name: path for path in data_root.rglob("*.zip")}
    output.mkdir(parents=True)
    splits, counts = {}, {}
    for name, entry in manifest["splits"].items():
        source = data / f"{name}.jsonl"
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError(f"{name}.jsonl does not match its manifest")
        rows = [json.loads(line) for line in payload.decode("utf-8").split("\n") if line.strip()]
        by_archive = {}
        for row in rows:
            by_archive.setdefault(row["_meta"]["source"]["archive"], []).append(row)
        for archive_name, items in by_archive.items():
            with zipfile.ZipFile(archives[archive_name]) as archive:
                for row in items:
                    raw = json.loads(archive.read(row["_meta"]["source"]["member"]).decode("utf-8-sig"))
                    if raw.get("recordId") != row["_meta"]["record_id"]:
                        raise ValueError("Source record changed since preparation")
                    subtype, triage = labels_for(raw, row["questions"]["dispatch"]["label"])
                    row["questions"]["subtype"] = {**AUX_QUESTIONS["subtype"], "label": subtype}
                    row["questions"]["triage"] = {**AUX_QUESTIONS["triage"], "label": triage}
        with (output / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[name] = {q: dict(Counter(r["questions"][q]["label"] for r in rows)) for q in AUX_QUESTIONS}
        splits[name] = {**entry, "labels": {**entry["labels"], **counts[name]},
                        "missing_labels": {**entry["missing_labels"], **{
                            q: sorted(set(AUX_QUESTIONS[q]["criteria"]) - set(counts[name][q])) for q in AUX_QUESTIONS}},
                        "sha256": hashlib.sha256((output / f"{name}.jsonl").read_bytes()).hexdigest()}
    augmented = {**manifest, "splits": splits, "original_splits": manifest["splits"],
                 "auxiliary": {"questions": list(AUX_QUESTIONS), "source_fields": ["disasterMedium", "triage"],
                               "role": "training targets only; never model input"}}
    if "parent_splits" in manifest:
        if parent is None:
            raise ValueError("This dataset has parent splits; pass the augmented parent with --parent")
        parent_manifest = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
        if parent_manifest.get("original_splits") != manifest["parent_splits"]:
            raise ValueError("--parent is not the augmentation of this dataset's parent")
        augmented["parent_splits"] = parent_manifest["splits"]
    write_json(output / "manifest.json", augmented)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="AI Hub 71768 label ZIP root")
    parser.add_argument("--data", type=Path, required=True, help="Prepared dataset to augment")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path, help="Augmented version of --data's parent dataset")
    args = parser.parse_args()
    print(json.dumps(augment(args.data_root, args.data, args.output, args.parent), ensure_ascii=False))


if __name__ == "__main__":
    main()
