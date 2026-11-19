import json
import sys
import zipfile

import pytest

from sentinel119.auxiliary import AUX_QUESTIONS, SUBTYPE_DISPATCH, augment, labels_for
from sentinel119.decision import read_examples
from sentinel119.prepare import prepare
from sentinel119.train import subtype_dispatch
from test_prepare import call

MEDIUM = {"구급": ("질병(중증)", "응급증상"), "화재": ("일반화재", ""), "구조": ("안전사고", ""), "기타": ("기타", "")}


def source(tmp_path, n=8):
    root = tmp_path / "source"
    root.mkdir()
    for prefix in ("TL", "VL"):
        with zipfile.ZipFile(root / f"{prefix}_synthetic.zip", "w") as archive:
            for i in range(n):
                dispatch = ("구급", "화재", "구조", "기타")[i % 4]
                raw = call(f"{prefix}-{i}", f"가상 {prefix} {i}", dispatch=dispatch, urgency=("상", "중", "하")[i % 3])
                raw["disasterMedium"], raw["triage"] = MEDIUM[dispatch]
                archive.writestr(f"{i}.json", json.dumps(raw, ensure_ascii=False))
    return root


def test_label_mapping_is_nested_and_complete():
    assert len(AUX_QUESTIONS["subtype"]["criteria"]) == 16 and len(AUX_QUESTIONS["triage"]["criteria"]) == 7
    assert set(SUBTYPE_DISPATCH.values()) == {"fire", "rescue", "ems", "other"}
    assert labels_for({"disasterLarge": "구급", "disasterMedium": "심정지", "triage": "사망추정"}, "ems") == (
        "ems_cardiac_arrest", "presumed_dead")
    assert labels_for({"disasterLarge": "화재", "disasterMedium": "산불", "triage": None}, "fire")[1] == "unrecorded"
    with pytest.raises(ValueError, match="nest"):
        labels_for({"disasterLarge": "구급", "disasterMedium": "부상", "triage": ""}, "fire")
    with pytest.raises(ValueError, match="Unknown subtype"):
        labels_for({"disasterLarge": "구급", "disasterMedium": "없는유형", "triage": ""}, "ems")
    # Derived dispatch sums probability mass within each dispatch type.
    keys = ["fire_general", "ems_illness", "ems_injury"]
    assert subtype_dispatch(keys, [0.4, 0.3, 0.3]) == "ems"


def test_augment_keeps_calls_and_adds_targets_not_inputs(tmp_path):
    root = source(tmp_path)
    data = tmp_path / "data"
    prepare(root, data, train_count=4, dev_count=2, test_count=2)
    out = tmp_path / "aug"
    counts = augment(root, data, out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for split in ("train", "dev", "test"):
        before = read_examples(data / f"{split}.jsonl")
        after = read_examples(out / f"{split}.jsonl", manifest["splits"][split]["sha256"])
        assert [e["_meta"]["group_id"] for e in before] == [e["_meta"]["group_id"] for e in after]
        assert [e["state"] for e in before] == [e["state"] for e in after]
        for e in after:
            assert set(e["questions"]) == {"dispatch", "urgency", "subtype", "triage"}
            assert SUBTYPE_DISPATCH[e["questions"]["subtype"]["label"]] == e["questions"]["dispatch"]["label"]
            assert "질병" not in e["state"] and "응급증상" not in e["state"]
    assert manifest["original_splits"] == json.loads((data / "manifest.json").read_text(encoding="utf-8"))["splits"]
    assert sum(counts["train"]["subtype"].values()) == 4
    with pytest.raises(ValueError, match="new output"):
        augment(root, data, out)


def test_child_dataset_requires_the_augmented_parent(tmp_path):
    root = source(tmp_path)
    parent, child = tmp_path / "parent", tmp_path / "child"
    # Different sizes so the parent's and child's splits genuinely differ.
    prepare(root, parent, train_count=3, dev_count=2, test_count=2)
    prepare(root, child, train_count=4, dev_count=2, test_count=2)
    manifest = json.loads((child / "manifest.json").read_text(encoding="utf-8"))
    manifest["parent_splits"] = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))["splits"]
    (child / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="--parent"):
        augment(root, child, tmp_path / "child-aug-a")
    parent_aug = tmp_path / "parent-aug"
    augment(root, parent, parent_aug)
    augment(root, child, tmp_path / "child-aug", parent=parent_aug)
    out = json.loads((tmp_path / "child-aug" / "manifest.json").read_text(encoding="utf-8"))
    assert out["parent_splits"] == json.loads((parent_aug / "manifest.json").read_text(encoding="utf-8"))["splits"]
    with pytest.raises(ValueError, match="not the augmentation"):
        augment(root, child, tmp_path / "child-aug-b", parent=tmp_path / "child-aug")


def test_cli_trains_with_auxiliary_questions(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer as FastTokenizer, models, pre_tokenizers
    from sentinel119.decision import SPECIAL
    from sentinel119.model import PointerHead, save_head
    from sentinel119.train import main
    from test_training import tiny_qwen

    model_dir = tmp_path / "tiny-model"
    model_dir.mkdir()
    torch.manual_seed(5)
    transformers.Qwen3_5ForConditionalGeneration(tiny_qwen(vision=True).config).save_pretrained(model_dir)
    vocab = {"[PAD]": 0, **{token: i + 1 for i, token in enumerate(SPECIAL)}, "[UNK]": 6}
    tokenizer = FastTokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                         pad_token="[PAD]").save_pretrained(model_dir)
    save_head(PointerHead(32, 8), model_dir / "decision_head.pt")
    root = source(tmp_path)
    data, aug = tmp_path / "data", tmp_path / "aug"
    prepare(root, data, train_count=4, dev_count=2, test_count=2)
    augment(root, data, aug)
    out = tmp_path / "fit"
    monkeypatch.setattr(sys, "argv", ["train", "--data", str(aug), "--model-dir", str(model_dir), "--device", "cpu",
        "--output", str(out), "--mode", "lora", "--max-steps", "2", "--grad-accum", "1", "--eval-every", "1",
        "--aux-weight", "0.5"])
    main()
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    assert run["objective"]["task_weights"] == {"dispatch": 1.0, "urgency": 1.0, "subtype": 0.5, "triage": 0.5}
    dev = run["result"]["adapted_raw_dev"]
    assert {"dispatch", "urgency", "subtype", "triage", "dispatch_from_subtype", "auxiliary_mean_nll"} <= set(dev)
    assert dev["dispatch_from_subtype"]["count"] == dev["dispatch"]["count"]
