import hashlib
import json
import sys
import zipfile

import pytest

from sentinel119.decision import read_examples
from sentinel119.hierarchy import QUESTIONS, allocate, prepare_hierarchy, symptom_group, targets
from sentinel119.prepare import prepare
from test_prepare import call

KINDS = [("구급", "질병(중증)", ["호흡곤란", "기타통증"]), ("구급", "임산부", []), ("구조", "자살", []),
         ("화재", "일반화재", []), ("기타", "기타", []), ("구급", "심정지", ["심정지"])]


def raw_call(record_id, i):
    large, medium, symptoms = KINDS[i % len(KINDS)]
    raw = call(record_id, f"가상 신고 {record_id}", dispatch=large, urgency=("상", "중", "하")[i % 3])
    raw.update({"disasterMedium": medium, "symptom": symptoms, "triage": ""})
    return raw


def source(tmp_path, n=36):
    root = tmp_path / "source"
    root.mkdir()
    for prefix in ("TL", "VL"):
        with zipfile.ZipFile(root / f"{prefix}_synthetic.zip", "w") as archive:
            for i in range(n):
                archive.writestr(f"{i}.json", json.dumps(raw_call(f"{prefix}-{i}", i), ensure_ascii=False))
    return root


def eval_dataset(tmp_path, root):
    """dev/confirm/test with real _meta, built from a small prepared dataset."""
    base = tmp_path / "base"
    prepare(root, base, train_count=6, dev_count=4, test_count=4)
    out = tmp_path / "eval"
    out.mkdir()
    rows = {name: (base / f"{name}.jsonl").read_bytes() for name in ("train", "dev", "test")}
    splits = {}
    for name, source_name in (("dev", "dev"), ("confirm", "train"), ("test", "test")):
        (out / f"{name}.jsonl").write_bytes(rows[source_name])
        splits[name] = {"calls": rows[source_name].count(b"\n"), "sha256": hashlib.sha256(rows[source_name]).hexdigest()}
    (out / "manifest.json").write_text(json.dumps({"splits": splits}), encoding="utf-8")
    return out


def test_targets_follow_the_hierarchy_and_symptom_priority():
    assert symptom_group(["기타통증", "호흡곤란"]) == "breathing"
    assert symptom_group(["심정지", "의식장애"]) == "arrest"
    assert symptom_group(["배뇨장애"]) == "other" and symptom_group([]) == "none" and symptom_group(None) == "none"
    assert targets({"disasterLarge": "구급", "disasterMedium": "임산부", "symptom": ["분만진통"]}) == {
        "ems_subtype": "pregnancy", "ems_symptom": "obstetric"}
    assert targets({"disasterLarge": "구조", "disasterMedium": "자살"}) == {"rescue_subtype": "suicide"}
    assert targets({"disasterLarge": "화재", "disasterMedium": "산불"}) == {}
    with pytest.raises(ValueError, match="Unknown EMS subtype"):
        targets({"disasterLarge": "구급", "disasterMedium": "자살"})
    assert len(QUESTIONS["ems_subtype"]["criteria"]) == 8 and len(QUESTIONS["rescue_subtype"]["criteria"]) == 4


def test_allocation_respects_floors_and_total():
    keys = ["a"] * 90 + ["b"] * 8 + ["c"] * 2
    quota = allocate(keys, 20, {"b": 6, "c": 5}, seed=0)
    assert sum(quota.values()) == 20 and quota["b"] >= 6 and quota["c"] == 2
    with pytest.raises(ValueError):
        allocate(keys, 5, {"a": 10}, seed=0)


def test_prepare_reuses_eval_calls_and_excludes_them_from_train(tmp_path):
    root = source(tmp_path)
    evaluation = eval_dataset(tmp_path, root)
    out = tmp_path / "hier"
    floors = {("구급", "임산부"): 3, ("구조", "자살"): 3}
    splits = prepare_hierarchy(root, evaluation, out, train_count=12, seed=3, floors=floors)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    train = read_examples(out / "train.jsonl", manifest["splits"]["train"]["sha256"])
    assert splits["train"]["calls"] == 12
    reserved = set()
    for name in ("dev", "confirm", "test"):
        before = [json.loads(l) for l in (evaluation / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l]
        after = read_examples(out / f"{name}.jsonl", manifest["splits"][name]["sha256"])
        assert [e["_meta"]["group_id"] for e in before] == [e["_meta"]["group_id"] for e in after]
        reserved |= {e["_meta"]["group_id"] for e in after}
    assert not reserved & {e["_meta"]["group_id"] for e in train}
    for e in train:
        d = e["questions"]["dispatch"]["label"]
        expected = {"ems": {"ems_subtype", "ems_symptom"}, "rescue": {"rescue_subtype"}}.get(d, set())
        assert set(e["questions"]) - {"dispatch", "urgency"} == expected
        assert "질병" not in e["state"] and "자살" not in e["state"]
    assert manifest["train_subtypes"].get("구급/임산부", 0) >= 3


def tiny_model(tmp_path, torch, transformers):
    from tokenizers import Tokenizer as FastTokenizer, models, pre_tokenizers
    from sentinel119.decision import SPECIAL
    from sentinel119.model import PointerHead, save_head
    from test_training import tiny_qwen
    model_dir = tmp_path / "tiny-model"
    model_dir.mkdir()
    torch.manual_seed(13)
    transformers.Qwen3_5ForConditionalGeneration(tiny_qwen(vision=True).config).save_pretrained(model_dir)
    vocab = {"[PAD]": 0, **{token: i + 1 for i, token in enumerate(SPECIAL)}, "[UNK]": 6}
    tok = FastTokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(model_dir)
    save_head(PointerHead(32, 8), model_dir / "decision_head.pt")
    return model_dir


def test_batched_evaluation_matches_single_call(tmp_path):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from sentinel119.model import DecisionModel, PointerHead
    from sentinel119.train import evaluate
    from test_training import Tokenizer, tiny_qwen
    root = source(tmp_path)
    evaluation = eval_dataset(tmp_path, root)
    out = tmp_path / "hier"
    prepare_hierarchy(root, evaluation, out, train_count=12, seed=3, floors={})
    examples = read_examples(out / "train.jsonl")
    torch.manual_seed(1)
    model = DecisionModel(tiny_qwen(vision=True), PointerHead(32, 8))
    single = evaluate(model, examples, Tokenizer(), 4096, calls_per_batch=1)
    batched = evaluate(model, examples, Tokenizer(), 4096, calls_per_batch=5)
    assert batched["mean_nll"] == pytest.approx(single["mean_nll"], abs=1e-4)
    for q in ("dispatch", "urgency", "ems_subtype"):
        assert batched[q]["confusion_matrix"] == single[q]["confusion_matrix"]
    assert "ems_subtype_end_to_end_accuracy" in single


def test_cli_hierarchy_questions_weights_and_filter(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from sentinel119.train import main
    root = source(tmp_path)
    evaluation = eval_dataset(tmp_path, root)
    data = tmp_path / "hier"
    prepare_hierarchy(root, evaluation, data, train_count=12, seed=3, floors={})
    model_dir = tiny_model(tmp_path, torch, transformers)
    common = ["train", "--data", str(data), "--model-dir", str(model_dir), "--device", "cpu"]
    full, control = tmp_path / "h1", tmp_path / "h0"
    monkeypatch.setattr(sys, "argv", common + ["--output", str(full), "--mode", "lora", "--max-steps", "2",
        "--grad-accum", "1", "--eval-every", "1", "--calls-per-batch", "3",
        "--task-weight", "ems_subtype=1.0", "--task-weight", "rescue_subtype=1.0", "--task-weight", "ems_symptom=0.3"])
    main()
    run = json.loads((full / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    assert run["objective"]["task_weights"]["ems_symptom"] == 0.3 and run["objective"]["task_weights"]["ems_subtype"] == 1.0
    dev = run["result"]["adapted_raw_dev"]
    assert {"ems_subtype", "rescue_subtype", "ems_symptom"} <= set(dev)
    monkeypatch.setattr(sys, "argv", common + ["--output", str(control), "--mode", "lora", "--max-steps", "1",
        "--grad-accum", "1", "--eval-every", "1", "--calls-per-batch", "2", "--questions", "dispatch,urgency"])
    main()
    run = json.loads((control / "run.json").read_text(encoding="utf-8"))
    assert set(run["objective"]["task_weights"]) == {"dispatch", "urgency"}
    assert not {"ems_subtype", "ems_symptom"} & set(run["result"]["adapted_raw_dev"])
    evaluated = tmp_path / "h0-confirm"
    monkeypatch.setattr(sys, "argv", common + ["--output", str(evaluated), "--evaluate-only", "confirm",
                                               "--checkpoint", str(control), "--calls-per-batch", "4"])
    main()
    metrics = json.loads((evaluated / "run.json").read_text(encoding="utf-8"))["result"]["metrics"]
    assert "ems_subtype" not in metrics  # checkpoint's own question set is reused
    with pytest.raises(SystemExit):
        monkeypatch.setattr(sys, "argv", common + ["--output", str(tmp_path / "bad"), "--task-weight", "x=-1"])
        main()
