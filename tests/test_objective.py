import json
import math
import random
import sys
import zipfile

import pytest

from sentinel119.prepare import make_example
from sentinel119.prompts import apply, extract_protocol, render
from sentinel119.train import learning_rate_scale, parse_multipliers, selection_key, training_order
from test_prepare import call

# Synthetic stand-in with the parsed layout; the real guideline stays outside the repository.
CHAPTER = """# 수보단계 중증응급환자 인지 프로토콜

<!-- p.156 -->
4. 무슨 일이 발생했는지 정확히 말해주시겠습니까?
※ 조기 긴급출동지시가 필요한 상황
○ 심정지
○ 심정지에 준한 상황
   ▶ 질식(Suffocation)
○ 6명 이상의 환자 발생
5. 심정지 상황 확인
5-1. 의식이 있습니까?     ⏏ 예  ⏏ 아니오
5-2. 정상적으로 숨 을 쉽니까?  ⏏ 예  ⏏ 아니오
6-1. ( 심인성 흉통 추정 ) 환자가 가슴 통증을 호소합니까?
6-2. ( 다발성/중증외상 추정)  6명 이상의 환자 ⏏ 예
   / 외상인 경우 : 환자가 의식이 없거나 숨을 쉬지 않습니까?  ⏏예
6-3. ( 아나필락시스 추정 ) 환자의 얼굴이 붓습니까?
6-4. ( 응급분만 추정 ) 태아가 보입니까?
6-5. ( 뇌졸중 의심 환자 추정 ) 팔다리 마비가 있습니까?
<!-- p.158 -->
5-2. 정상적으로 숨을 쉽니까? ⏏ 예
"""


@pytest.fixture
def guideline(tmp_path):
    chapters = tmp_path / "knowledge" / "extracted" / "chapters"
    chapters.mkdir(parents=True)
    (chapters / "p2-s2-c02_수보단계_중증응급환자_인지_프로토콜.md").write_text(CHAPTER, encoding="utf-8")
    return tmp_path / "knowledge"


def test_protocol_parse_prefers_clean_repeats_and_rejects_other_layouts():
    parsed = extract_protocol(CHAPTER)
    assert parsed["early"] == ["심정지", "심정지에 준한 상황", "질식(Suffocation)", "6명 이상의 환자 발생"]
    assert parsed["vitals"] == ["의식이 있습니까?", "정상적으로 숨을 쉽니까?"]
    assert parsed["special"][1] == ("다발성/중증외상 추정", "환자가 의식이 없거나 숨을 쉬지 않습니까?")
    with pytest.raises(ValueError, match="layout"):
        extract_protocol(CHAPTER.replace("6-5.", "7-5."))


def test_prompt_changes_only_choice_text_and_records_source(guideline):
    example = make_example(call(), "group", {})
    assert render("bare") == {"variant": "bare", "overrides": {}}
    with pytest.raises(ValueError, match="guideline-dir"):
        render("protocol")
    spec = render("protocol-checklist", guideline)
    assert spec["source"]["sha256"] and spec["source"]["label_alignment"] == "unverified hypothesis"
    changed = apply(example, spec)
    urgency = changed["questions"]["urgency"]
    assert urgency["criteria"]["high"].startswith("상 — ") and "심인성 흉통" in urgency["criteria"]["high"]
    assert urgency["criteria"]["medium"] == "중" and urgency["criteria"]["low"] == "하"
    assert "정상적으로 숨을 쉽니까?" in urgency["instructions"]
    assert urgency["label"] == example["questions"]["urgency"]["label"]
    assert changed["questions"]["dispatch"] == example["questions"]["dispatch"]
    assert example["questions"]["urgency"]["criteria"]["high"] == "상"
    with pytest.raises(ValueError, match="adds choices"):
        apply(example, {"overrides": {"urgency": {"criteria": {"critical": "x"}}}})


def rows_for(labels_and_keys):
    return [{"qid": qid, "keys": keys, "label": keys.index(label)} for qid, keys, label in labels_and_keys]


def test_training_loss_defaults_to_nll_and_each_term_acts():
    torch = pytest.importorskip("torch")
    from sentinel119.model import supervised_loss, training_loss, urgency_cost
    keys = ["high", "medium", "low"]
    rows = rows_for([("dispatch", ["ems", "fire", "rescue", "other"], "ems"), ("urgency", keys, "high")])
    logits = (torch.tensor([2.0, 0.0, 0.0, -1.0]), torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(training_loss(logits, rows), supervised_loss(logits, rows))
    assert urgency_cost("high", "low", 3.0) == 6.0 and urgency_cost("low", "high", 3.0) == 2.0
    costed = training_loss(logits, rows, {"cost": {"urgency": {"weight": 1.0, "downward": 3.0}}})
    assert costed > supervised_loss(logits, rows)
    # The cost term penalizes mass on "low" more than equal mass on "medium" when the truth is "high".
    toward_low = training_loss((logits[0], torch.tensor([0.0, 0.0, 1.0])), rows,
                               {"cost": {"urgency": {"weight": 1.0, "downward": 3.0}}, "task_weights": {"dispatch": 0}})
    toward_mid = training_loss((logits[0], torch.tensor([0.0, 1.0, 0.0])), rows,
                               {"cost": {"urgency": {"weight": 1.0, "downward": 3.0}}, "task_weights": {"dispatch": 0}})
    assert toward_low > toward_mid
    weighted = training_loss(logits, rows, {"task_weights": {"urgency": 3.0}})
    per_q = [float(-torch.log_softmax(z, -1)[r["label"]]) for z, r in zip(logits, rows)]
    assert float(weighted) == pytest.approx((per_q[0] + 3 * per_q[1]) / 4)
    smoothed = training_loss(logits, rows, {"label_smoothing": {"urgency": 0.1}})
    assert float(smoothed) != pytest.approx(float(supervised_loss(logits, rows)))
    # Logit adjustment with a tiny "other" prior raises the loss of predicting the majority class.
    adjust = {"logit_adjust": {"dispatch": {"ems": 0.0, "fire": -2.0, "rescue": -2.0, "other": -4.0}}}
    assert training_loss(logits, rows, adjust) < supervised_loss(logits, rows) + 1e-6


def test_schedule_selection_and_oversampling():
    assert learning_rate_scale(1, 10) == 1.0
    assert learning_rate_scale(1, 10, 0.2, "cosine") == pytest.approx(0.5)
    assert learning_rate_scale(2, 10, 0.2, "cosine") == pytest.approx(1.0)
    assert learning_rate_scale(10, 10, 0.2, "cosine") == pytest.approx(0.0, abs=1e-12)
    metrics = lambda f1, down, nll: {"mean_nll": nll, "dispatch": {"macro_f1": f1}, "urgency":
                                     {"macro_f1": f1, "downward_error_rate": down}}
    good, risky = metrics(0.6, 0.1, 0.9), metrics(0.8, 0.4, 0.5)
    assert selection_key(risky) > selection_key(good)
    assert selection_key(risky, "composite") > selection_key(good, "composite")
    assert selection_key(good, "composite", 0.2) > selection_key(risky, "composite", 0.2)
    examples = [make_example(call(), f"g{i}", {}) for i in range(3)]
    examples[0]["questions"]["dispatch"]["label"] = "other"
    order = training_order(examples, {"other": 3}, random.Random(0))
    assert sorted(order) == [0, 0, 0, 1, 2]
    assert parse_multipliers("other=3,rescue=2") == {"other": 3, "rescue": 2}
    for bad in ("other=0", "unknown=2", "other"):
        with pytest.raises(ValueError):
            parse_multipliers(bad)


def test_cli_prompt_objective_and_checkpoint_prompt_reuse(tmp_path, monkeypatch, guideline):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer as FastTokenizer, models, pre_tokenizers
    from sentinel119.decision import SPECIAL
    from sentinel119.model import PointerHead, save_head
    from sentinel119.prepare import prepare
    from sentinel119.train import main
    from test_training import tiny_qwen

    model_dir = tmp_path / "tiny-model"
    model_dir.mkdir()
    torch.manual_seed(11)
    transformers.Qwen3_5ForConditionalGeneration(tiny_qwen(vision=True).config).save_pretrained(model_dir)
    vocab = {"[PAD]": 0, **{token: i + 1 for i, token in enumerate(SPECIAL)}, "[UNK]": 6}
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
    fit_dir, eval_dir = tmp_path / "fit", tmp_path / "eval"
    common = ["train", "--data", str(data), "--model-dir", str(model_dir), "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", common + ["--output", str(fit_dir), "--mode", "lora",
        "--lora-targets", "attention-mlp", "--max-steps", "3", "--grad-accum", "1", "--eval-every", "1",
        "--prompt-variant", "protocol-checklist", "--guideline-dir", str(guideline),
        "--urgency-weight", "1.5", "--urgency-label-smoothing", "0.1", "--urgency-cost-weight", "0.5",
        "--dispatch-logit-adjust", "1.0", "--oversample", "other=3", "--head-learning-rate", "1e-3",
        "--warmup-ratio", "0.34", "--schedule", "cosine", "--select-by", "composite",
        "--max-downward-rate", "0.5", "--keep-checkpoints"])
    main()
    trained = json.loads((fit_dir / "run.json").read_text(encoding="utf-8"))
    assert trained["status"] == "completed" and trained["prompt"]["variant"] == "protocol-checklist"
    assert trained["objective"]["cost"]["urgency"] == {"weight": 0.5, "downward": 2.0}
    assert all(math.isfinite(v) for v in trained["objective"]["logit_adjust"]["dispatch"].values())
    assert any(".mlp." in name for name in trained["trainable_names"])
    assert trained["result"]["selection"]["rule"] == "composite"
    assert sorted(p.name for p in (fit_dir / "checkpoints").iterdir()) == ["step-1", "step-2", "step-3"]
    history = [json.loads(line) for line in (fit_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [round(e["lr_scale"], 6) for e in history] == [1.0, 0.5, 0.0]
    monkeypatch.setattr(sys, "argv", common + ["--output", str(eval_dir), "--evaluate-only", "test",
        "--checkpoint", str(fit_dir)])
    main()
    evaluated = json.loads((eval_dir / "run.json").read_text(encoding="utf-8"))
    assert evaluated["status"] == "completed" and evaluated["prompt"] == trained["prompt"]
    monkeypatch.setattr(sys, "argv", common + ["--output", str(tmp_path / "mismatch"), "--evaluate-only",
        "test", "--checkpoint", str(fit_dir), "--prompt-variant", "bare"])
    with pytest.raises(ValueError, match="prompt"):
        main()
