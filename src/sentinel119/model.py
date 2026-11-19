"""Zoro decision head adaptation, optionally with text-attention LoRA."""

from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from .decision import MODEL_ID, MODEL_REVISION


class PointerHead(nn.Module):
    """Zoro/Kev pointer head; parameter names match the released head (see notices)."""

    def __init__(self, hidden_size, head_dim=256):
        super().__init__()
        self.q = nn.Linear(hidden_size, head_dim)
        self.k = nn.Linear(hidden_size, head_dim)
        self.scale = 1 / math.sqrt(head_dim)

    def forward(self, decisions, options, owner):
        return (self.k(options) * self.q(decisions)[owner]).sum(-1) * self.scale


def load_head(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    head = PointerHead(saved["head"]["q.weight"].shape[1], saved["head_dim"])
    head.load_state_dict(saved["head"], strict=True)
    return head.to(device=device, dtype=torch.float32)


def save_head(head, path):
    torch.save({"head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                "head_dim": head.q.out_features}, path)


class DecisionModel(nn.Module):
    def __init__(self, backbone, head, pad_id=0):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.head = head.float()
        self.pad_id = pad_id
        self.train_backbone = False

    def features(self, rows):
        device = next(self.backbone.parameters()).device
        length = max(len(row["ids"]) for row in rows)
        ids = torch.full((len(rows), length), self.pad_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, :len(row["ids"])] = torch.tensor(row["ids"], device=device)
            mask[i, :len(row["ids"])] = 1
        if not self.train_backbone:
            self.backbone.eval()
        # no_grad, not inference_mode: detached features are still needed by head backward.
        with nullcontext() if self.train_backbone else torch.no_grad():
            hidden = self.backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            decisions = torch.stack([hidden[i, row["decide"]] for i, row in enumerate(rows)]).float()
            options = torch.cat([hidden[i, row["opts"]] for i, row in enumerate(rows)]).float()
        owner = torch.tensor([i for i, row in enumerate(rows) for _ in row["opts"]], device=device)
        return decisions, options, owner

    def forward(self, rows):
        logits = self.head(*self.features(rows))
        return logits.split([len(row["opts"]) for row in rows])


LORA_TARGETS = {
    # Text decoder only; the vision tower and gated-delta (linear_attn) projections stay frozen.
    "attention-qv": r"(?:language_model\.)?layers\.\d+\.self_attn\.(?:q_proj|v_proj)",
    "attention": r"(?:language_model\.)?layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)",
    "attention-mlp": (r"(?:language_model\.)?layers\.\d+\."
                      r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"),
}


def enable_lora(model, rank=8, alpha=16, targets="attention-qv"):
    from peft import LoraConfig, get_peft_model
    if targets not in LORA_TARGETS:
        raise ValueError(f"Unknown LoRA target set {targets}")
    targets = LORA_TARGETS[targets]
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0,
                        target_modules=targets, bias="none")
    model.backbone = get_peft_model(model.backbone, config)
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train_backbone = True


def save_adaptation(model, output):
    output.mkdir(parents=True, exist_ok=True)
    save_head(model.head, output / "decision_head.pt")
    if model.train_backbone:
        model.backbone.save_pretrained(output / "adapter", safe_serialization=True, save_embedding_layers=False)


def reload_adaptation(model, output):
    model.head = load_head(output / "decision_head.pt", next(model.head.parameters()).device)
    if model.train_backbone:
        from peft import PeftModel
        base = model.backbone.unload()
        model.backbone = PeftModel.from_pretrained(base, output / "adapter", is_trainable=True)


def same_base(left, right):
    fields = ("model_id", "revision", "local_override", "config_sha256", "base_head_sha256")
    if left.get("local_override") or right.get("local_override"):
        fields += ("local_model_dir",)
    return all(left.get(field) == right.get(field) for field in fields)


def supervised_loss(logits, rows):
    """Plain mean NLL; used for evaluation and NLL-based selection regardless of the training objective."""
    return torch.stack([-torch.log_softmax(z.float(), dim=-1)[row["label"]]
                        for z, row in zip(logits, rows)]).mean()


SEVERITY = {"low": 0, "medium": 1, "high": 2}


def urgency_cost(actual, predicted, downward_cost):
    """Asymmetric ordinal cost: upward errors cost their distance, downward errors downward_cost per level."""
    gap = SEVERITY[predicted] - SEVERITY[actual]
    return float(gap) if gap >= 0 else downward_cost * -gap


def training_loss(logits, rows, objective=None):
    """Weighted per-question training objective; with objective=None it equals supervised_loss.

    objective keys (all optional, defaults reproduce plain CE):
      task_weights {qid: w}; label_smoothing {qid: eps};
      logit_adjust {qid: {key: tau*log(prior)}} added to logits during training only (Menon et al., 2021);
      cost {qid: {"weight": lam, "downward": c}} adds lam * E_p[cost(label, k)] for ordinal urgency.
    """
    objective = objective or {}
    terms, weights = [], []
    for z, row in zip(logits, rows):
        qid, keys = row["qid"], row["keys"]
        z = z.float()
        if adjust := objective.get("logit_adjust", {}).get(qid):
            z = z + torch.tensor([adjust[k] for k in keys], device=z.device)
        log_p = torch.log_softmax(z, dim=-1)
        eps = objective.get("label_smoothing", {}).get(qid, 0.0)
        loss = -(1 - eps) * log_p[row["label"]] - eps * log_p.mean()
        if cost := objective.get("cost", {}).get(qid):
            actual = keys[row["label"]]
            matrix = torch.tensor([urgency_cost(actual, k, cost["downward"]) for k in keys], device=z.device)
            loss = loss + cost["weight"] * (log_p.exp() * matrix).sum()
        terms.append(loss)
        weights.append(objective.get("task_weights", {}).get(qid, 1.0))
    weights = torch.tensor(weights, device=terms[0].device)
    return (torch.stack(terms) * weights).sum() / weights.sum()


def load_zoro(*, device="cuda", model_dir=None, head_path=None, adapter_path=None, local_files_only=False):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use a GPU environment (or explicitly --device cpu)")
    path = Path(model_dir) if model_dir else Path(snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=local_files_only,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "decision_head.pt"]))
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_5":
        raise ValueError("Expected the current Qwen3.5 Zoro-1 checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    if device.startswith("cuda") and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This training path requires BF16-capable CUDA hardware")
    lm = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=dtype, local_files_only=True)
    backbone = lm.model.to(device)
    head = load_head(head_path or path / "decision_head.pt", device)
    if head.q.in_features != config["text_config"]["hidden_size"]:
        raise ValueError("Decision head and backbone hidden sizes differ")
    model = DecisionModel(backbone, head, tokenizer.pad_token_id or 0)
    if adapter_path:
        from peft import PeftModel
        model.backbone = PeftModel.from_pretrained(model.backbone, adapter_path, is_trainable=False)
    identity = {"model_id": MODEL_ID, "revision": MODEL_REVISION if model_dir is None else None,
                "local_model_dir": str(path.resolve()), "local_override": model_dir is not None,
                "dtype": str(dtype),
                "config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
                "base_head_sha256": hashlib.sha256((path / "decision_head.pt").read_bytes()).hexdigest()}
    return model, tokenizer, identity
