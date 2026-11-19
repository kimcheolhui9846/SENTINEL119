import importlib.util
import json
from pathlib import Path
import random
import sys

import pytest

from sentinel119.decision import read_examples
from sentinel119.hierarchy import prepare_hierarchy
from test_hierarchy import eval_dataset, source, tiny_model

spec = importlib.util.spec_from_file_location("continuation_split",
                                              Path(__file__).parents[1] / "tools" / "continuation_split.py")
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)


def hier(tmp_path):
    root = source(tmp_path)
    data = tmp_path / "hier"
    prepare_hierarchy(root, eval_dataset(tmp_path, root), data, train_count=12, seed=3, floors={})
    return data


def test_continuation_keeps_unseen_calls_in_epoch_order(tmp_path):
    data = hier(tmp_path)
    out = tmp_path / "cont"
    assert cs.continuation(data, out, seed=119, consumed=5) == 7
    train = read_examples(data / "train.jsonl")
    order = list(range(len(train))); random.Random(119).shuffle(order)
    rest = read_examples(out / "train.jsonl")
    assert [e["_meta"]["group_id"] for e in rest] == [train[i]["_meta"]["group_id"] for i in order[5:]]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["continuation"]["consumed_calls"] == 5
    for name in ("dev", "confirm", "test"):
        assert (out / f"{name}.jsonl").read_bytes() == (data / f"{name}.jsonl").read_bytes()
    with pytest.raises(ValueError):
        cs.continuation(data, tmp_path / "bad", seed=119, consumed=12)


def test_init_checkpoint_starts_from_saved_weights(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from sentinel119.train import main
    data = hier(tmp_path)
    cont = tmp_path / "cont"; cs.continuation(data, cont, seed=119, consumed=4)
    model_dir = tiny_model(tmp_path, torch, transformers)
    common = ["train", "--model-dir", str(model_dir), "--device", "cpu", "--mode", "lora", "--grad-accum", "1",
              "--eval-every", "1", "--learning-rate", "1e-3"]
    first, second = tmp_path / "first", tmp_path / "second"
    monkeypatch.setattr(sys, "argv", common + ["--data", str(data), "--output", str(first), "--max-steps", "2"])
    main()
    monkeypatch.setattr(sys, "argv", common + ["--data", str(cont), "--output", str(second), "--max-steps", "1",
                                               "--init-checkpoint", str(first)])
    main()
    a = json.loads((first / "run.json").read_text(encoding="utf-8"))
    b = json.loads((second / "run.json").read_text(encoding="utf-8"))
    assert b["status"] == "completed" and b["init_checkpoint"]["best_step"] == a["result"]["best_step"]
    # Same dev calls: the continued run's pre-training dev score equals the source run's selected dev score.
    assert b["result"]["base_raw_dev"]["mean_nll"] == pytest.approx(a["result"]["adapted_raw_dev"]["mean_nll"], abs=1e-5)
    assert any("lora_" in n for n in b["trainable_names"])
    monkeypatch.setattr(sys, "argv", common + ["--data", str(cont), "--output", str(tmp_path / "x"),
                                               "--evaluate-only", "dev", "--init-checkpoint", str(first)])
    with pytest.raises(SystemExit):
        main()
