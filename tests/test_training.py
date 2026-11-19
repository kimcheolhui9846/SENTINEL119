from copy import deepcopy
import json
import random
import sys
from types import SimpleNamespace
import zipfile

import pytest

from sentinel119.decision import ContextOverflow, SPECIAL, encode, read_examples
from sentinel119.prepare import make_example
from sentinel119.train import classification_metrics, fit
from test_prepare import call


class Tokenizer:
    pad_token_id = 0

    def get_vocab(self):
        return {token: i + 1 for i, token in enumerate(SPECIAL)}

    def convert_tokens_to_ids(self, token):
        return self.get_vocab()[token]

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[6 + ord(c) % 80 for c in text])


def test_encoding_label_independence_permutation_and_no_truncation():
    example = make_example(call(), "group", {})
    first = encode(example, Tokenizer())
    changed = deepcopy(example)
    changed["questions"]["dispatch"]["label"] = "fire"
    assert first[0]["ids"] == encode(changed, Tokenizer())[0]["ids"]
    for seed in range(8):
        rows = encode(example, Tokenizer(), rng=random.Random(seed))
        assert rows[0]["keys"][rows[0]["label"]] == "ems"
        assert rows[1]["keys"][rows[1]["label"]] == "high"
    with pytest.raises(ContextOverflow):
        encode(example, Tokenizer(), max_length=10)


def test_dataset_hash_and_prefix_scope_are_enforced(tmp_path):
    path = tmp_path / "data.jsonl"
    example = make_example(call(), "group", {})
    path.write_text(json.dumps(example), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        read_examples(path, "wrong")
    example["_meta"]["label_scope"] = "unreviewed_prefix"
    path.write_text(json.dumps(example), encoding="utf-8")
    with pytest.raises(ValueError, match="full-call"):
        read_examples(path)


def test_metrics_include_missing_classes():
    result = classification_metrics(["high", "low"], ["high", "high"], ["high", "medium", "low"])
    assert result["accuracy"] == 0.5
    assert result["macro_f1"] == pytest.approx(2 / 9)
    assert result["missing_labels"] == ["medium"]


def test_fit_updates_only_head_and_reloads(tmp_path):
    torch = pytest.importorskip("torch")
    from torch import nn
    from sentinel119.model import DecisionModel, PointerHead

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 16)

        def forward(self, input_ids, attention_mask, use_cache=False):
            hidden = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
            return SimpleNamespace(last_hidden_state=hidden.cumsum(1) / 20)

    torch.manual_seed(42)
    model = DecisionModel(Backbone(), PointerHead(16, 8))
    original_backbone = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    original_head = {k: v.clone() for k, v in model.head.state_dict().items()}
    example = make_example(call(), "group", {})
    result = fit(model, Tokenizer(), [example], [example], tmp_path,
                 max_steps=3, grad_accum=2, eval_every=2)
    assert result["nonzero_head_gradient"] and result["reload_verified"]
    assert all(torch.equal(v, original_backbone[k]) for k, v in model.backbone.state_dict().items())
    assert any(not torch.equal(v, original_head[k]) for k, v in model.head.state_dict().items())


def tiny_qwen(vision=False):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3_5TextConfig(vocab_size=100, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2,
        linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 0]})
    if vision:
        vl_config = transformers.Qwen3_5Config(text_config=config.to_dict(), vision_config={
            "depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
            "out_hidden_size": 32, "num_position_embeddings": 16, "patch_size": 2,
            "spatial_merge_size": 2, "temporal_patch_size": 2, "deepstack_visual_indexes": []})
        return transformers.Qwen3_5Model(vl_config)
    return transformers.Qwen3_5TextModel(config)


def test_tiny_qwen35_causal_backbone_gradient_contract():
    torch = pytest.importorskip("torch")
    from sentinel119.model import DecisionModel, PointerHead, supervised_loss
    model = DecisionModel(tiny_qwen(vision=True), PointerHead(32, 8))
    rows = [{"ids": [1, 5, 7, 8, 9], "decide": 4, "opts": [1, 3], "label": 0},
            {"ids": [2, 4, 6], "decide": 2, "opts": [0, 1], "label": 1}]
    logits = model(rows)
    single = model([rows[1]])[0]
    torch.testing.assert_close(logits[1], single, atol=1e-5, rtol=1e-5)
    loss = supervised_loss(logits, rows)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.head.q.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.parameters())


def test_lora_fit_updates_adapters_and_reloads(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from sentinel119.model import DecisionModel, PointerHead, enable_lora
    torch.manual_seed(7)
    model = DecisionModel(tiny_qwen(vision=True), PointerHead(32, 8))
    enable_lora(model, rank=2, alpha=4)
    frozen = {k: p.detach().clone() for k, p in model.backbone.named_parameters() if not p.requires_grad}
    example = make_example(call(), "group", {})
    result = fit(model, Tokenizer(), [example], [example], tmp_path,
                 max_steps=2, grad_accum=1, eval_every=1, learning_rate=1e-3)
    assert result["nonzero_adapter_gradient"] and result["reload_verified"]
    assert (tmp_path / "adapter" / "adapter_model.safetensors").is_file()
    for name, p in model.backbone.named_parameters():
        if name in frozen:
            assert torch.equal(p, frozen[name])
    assert len(frozen.keys() & dict(model.backbone.named_parameters()).keys()) == len(frozen)


def test_pinned_checkpoint_identity_is_cache_location_independent():
    pytest.importorskip("torch")
    from sentinel119.model import same_base
    base = {"model_id": "zoro", "revision": "pinned", "local_override": False,
            "config_sha256": "config", "base_head_sha256": "head", "local_model_dir": "old/cache"}
    assert same_base(base, {**base, "local_model_dir": "new/cache", "dtype": "bf16"})
    assert not same_base(base, {**base, "revision": "different"})


@pytest.mark.parametrize("mode", ["lora", "head-only"])
def test_cli_train_and_independent_checkpoint_evaluation(tmp_path, monkeypatch, mode):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer as FastTokenizer, models, pre_tokenizers
    from sentinel119.model import PointerHead, save_head
    from sentinel119.prepare import prepare
    from sentinel119.train import main

    model_dir = tmp_path / "tiny-model"
    model_dir.mkdir()
    torch.manual_seed(9)
    lm = transformers.Qwen3_5ForConditionalGeneration(tiny_qwen(vision=True).config)
    lm.save_pretrained(model_dir)
    vocab = {"[PAD]": 0, **{token: i + 1 for i, token in enumerate(SPECIAL)}, "[UNK]": 6,
             "ems": 7, "fire": 8, "rescue": 9, "other": 10, "high": 11, "medium": 12, "low": 13}
    tokenizer = FastTokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer,
        unk_token="[UNK]", pad_token="[PAD]").save_pretrained(model_dir)
    save_head(PointerHead(32, 8), model_dir / "decision_head.pt")
    source = tmp_path / "source"
    source.mkdir()
    for prefix in ("TL", "VL"):
        with zipfile.ZipFile(source / f"{prefix}_synthetic.zip", "w") as archive:
            for i in range(8):
                archive.writestr(f"{i}.json", json.dumps(call(f"{prefix}-{i}", f"가상 {prefix} {i}")))
    data = tmp_path / "dataset"
    prepare(source, data, train_count=4, dev_count=2, test_count=2)
    train_output, eval_output = tmp_path / "fit", tmp_path / "eval"
    common = ["train", "--data", str(data), "--model-dir", str(model_dir), "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", common + ["--output", str(train_output), "--mode", mode,
        "--max-steps", "2", "--grad-accum", "1", "--eval-every", "1", "--keep-checkpoints"])
    main()
    trained = json.loads((train_output / "run.json").read_text(encoding="utf-8"))
    assert trained["status"] == "completed" and trained["mode"] == mode
    assert trained["result"]["nonzero_adapter_gradient"] == (mode == "lora")
    if mode == "head-only":
        assert all(k.startswith("head.") for k in trained["trainable_names"])
        assert not (train_output / "adapter").exists()
        assert (train_output / "checkpoints/step-1/decision_head.pt").is_file()
    monkeypatch.setattr(sys, "argv", common + ["--output", str(eval_output), "--evaluate-only", "test",
        "--checkpoint", str(train_output)])
    main()
    evaluated = json.loads((eval_output / "run.json").read_text(encoding="utf-8"))
    assert evaluated["status"] == "completed" and evaluated["mode"] == mode
    assert evaluated["result"]["split"] == "test"
    assert evaluated["result"]["metrics"]["dispatch"]["count"] == 2
