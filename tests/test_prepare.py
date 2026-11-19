import json
from pathlib import Path
import zipfile

import pytest

from sentinel119.prepare import Entry, Groups, eligible_groups, make_example, normalize_call, prepare, stratified_take


def call(record_id="call-1", text="도움이 필요해요.", dispatch="구급", urgency="상"):
    return {"recordId": record_id, "disasterLarge": dispatch, "urgencyLevel": urgency,
            "address": "META_ADDRESS_MUST_NOT_APPEAR", "symptom": ["META_SYMPTOM"],
            "utterances": [{"id": "turn-2", "startAt": 200, "endAt": 300, "speaker": 1, "text": text},
                           {"id": "turn-1", "startAt": 0, "endAt": 100, "speaker": 0, "text": "119입니다."}]}


def test_metadata_does_not_leak_and_turns_are_chronological():
    example = make_example(call(), "group", {})
    assert example["state"] == "화자0: 119입니다.\n화자1: 도움이 필요해요."
    assert "META_" not in example["state"]
    assert example["questions"]["dispatch"]["label"] == "ems"
    assert [t["id"] for t in example["_meta"]["turns"]] == ["turn-1", "turn-2"]
    assert [t["source_index"] for t in example["_meta"]["turns"]] == [1, 0]
    assert normalize_call(call())["reordered"]


@pytest.mark.parametrize("field,value,error", [("speaker", 2, "unknown_speaker"),
    ("text", "  ", "empty_turn"), ("startAt", float("nan"), "invalid_time"), ("endAt", -1, "invalid_time")])
def test_bad_turns_are_rejected(field, value, error):
    raw = call()
    raw["utterances"][0][field] = value
    with pytest.raises(ValueError, match=error):
        normalize_call(raw)


def test_transitive_record_and_content_overlap_is_quarantined():
    entries = [Entry(Path("x"), str(i), split, rid, content, ("ems", "high"))
               for i, (split, rid, content) in enumerate([
                   ("Training", "A", "X"), ("Training", "A", "Y"), ("Validation", "B", "Y")])]
    groups = Groups()
    for e in entries:
        groups.add(e.record_id, e.content_hash)
    pools, report = eligible_groups(entries, groups)
    assert not pools["Training"] and not pools["Validation"]
    assert report["cross_official_records_excluded"] == 3


def test_conflicting_duplicate_labels_are_excluded():
    entries = [Entry(Path("x"), str(i), "Training", "A", "X", ("ems", label))
               for i, label in enumerate(["high", "low"])]
    groups = Groups()
    for e in entries:
        groups.add(e.record_id, e.content_hash)
    pools, report = eligible_groups(entries, groups)
    assert pools["Training"] == []
    assert report["conflicting_label_groups_excluded"] == 1


def test_small_stratified_sample_covers_marginal_classes():
    pool = [(str(i), Entry(Path("x"), str(i), "Training", str(i), str(i), (dispatch, urgency)))
            for i, (dispatch, urgency) in enumerate(
                (d, u) for d in ("ems", "fire", "rescue", "other") for u in ("low", "medium", "high"))]
    selected, _ = stratified_take(pool, 10, 119)
    assert {e.labels[0] for _, e in selected} == {"ems", "fire", "rescue", "other"}
    assert {e.labels[1] for _, e in selected} == {"low", "medium", "high"}


def test_zip_pipeline_deterministic_and_split_isolated(tmp_path):
    for prefix in ("TL", "VL"):
        with zipfile.ZipFile(tmp_path / f"{prefix}_synthetic.zip", "w") as archive:
            for i in range(24):
                raw = call(f"{prefix}-{i}", f"가상 사례 {prefix} {i}",
                           ["화재", "구조", "구급", "기타"][i % 4], ["상", "중", "하"][i % 3])
                archive.writestr(f"{i}.json", json.dumps(raw, ensure_ascii=False))
    a, b = tmp_path / "a", tmp_path / "b"
    report = prepare(tmp_path, a, train_count=12, dev_count=12, test_count=12)
    prepare(tmp_path, b, train_count=12, dev_count=12, test_count=12)
    assert report["splits"]["train"]["missing_labels"] == {"dispatch": [], "urgency": []}
    ids = []
    for split in ("train", "dev", "test"):
        assert (a / f"{split}.jsonl").read_bytes() == (b / f"{split}.jsonl").read_bytes()
        rows = [json.loads(line) for line in (a / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        ids.append({r["_meta"]["group_id"] for r in rows})
        assert all(r["_meta"]["label_scope"] == "full_call" for r in rows)
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    with pytest.raises(ValueError, match="empty"):
        prepare(tmp_path, a)
