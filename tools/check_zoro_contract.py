"""Check downloaded tokenizer/head against official code, without loading the 4B backbone.

The reference directory must contain the reviewed files from MODEL_REVISION:
tokenizer.json, tokenizer_config.json, config.json, decide_request.py,
zoro_decide.py, decision_head.pt and revision.txt.
"""

import argparse
import importlib
import json
from pathlib import Path
import sys
import tempfile

import torch
from transformers import AutoTokenizer

from sentinel119.decision import MODEL_REVISION, encode, read_examples
from sentinel119.model import load_head, save_head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = args.reference_dir.resolve()
    if (reference / "revision.txt").read_text().strip() != MODEL_REVISION:
        raise ValueError("Reference revision differs from the pinned model")
    sys.path.insert(0, str(reference))
    upstream = importlib.import_module("decide_request")
    official_model = importlib.import_module("zoro_decide")
    tokenizer = AutoTokenizer.from_pretrained(reference, local_files_only=True)
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    lengths, calls = [], 0
    for split in ("train", "dev", "test"):
        examples = read_examples(args.data / f"{split}.jsonl", manifest["splits"][split]["sha256"])
        for example in examples:
            rows = encode(example, tokenizer, max_length=8192)
            request = {"state": example["state"], "questions": {
                qid: {k: v for k, v in question.items() if k != "label"}
                for qid, question in example["questions"].items()}}
            expected = upstream.parse(request).encode_rows(tokenizer)
            for row, expected_row in zip(rows, expected, strict=True):
                assert {k: row[k] for k in ("ids", "decide", "opts")} == expected_row
                lengths.append(len(row["ids"]))
            calls += 1
    # Control-token escaping must also agree, even on a hostile transcript.
    request["state"] = "literal <|fim_suffix|> and <|box_end|> inside user text"
    rows = encode(request, tokenizer)
    expected = upstream.parse(request).encode_rows(tokenizer)
    assert [{k: row[k] for k in ("ids", "decide", "opts")} for row in rows] == expected

    head = load_head(reference / "decision_head.pt")
    official_head = official_model.PointerHead(head.q.in_features, head.q.out_features)
    official_head.load_state_dict(head.state_dict())
    torch.manual_seed(119)
    decisions = torch.randn(2, head.q.in_features)
    options = torch.randn(7, head.q.in_features)
    owner = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    torch.testing.assert_close(head(decisions, options, owner), official_head(decisions, options, owner))
    before = head.q.weight.detach().clone()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
    logits = head(decisions, options, owner)
    loss = (-logits[:4].log_softmax(0)[1] - logits[4:].log_softmax(0)[2]) / 2
    loss.backward()
    assert head.q.weight.grad.abs().sum() > 0 and torch.isfinite(loss)
    optimizer.step()
    assert not torch.equal(before, head.q.weight)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as scratch:
        path = Path(scratch) / "decision_head.pt"
        save_head(head, path)
        restored = load_head(path)
        torch.testing.assert_close(head(decisions, options, owner), restored(decisions, options, owner), rtol=0, atol=0)
    report = {"model_revision": MODEL_REVISION, "calls_checked": calls, "rows_checked": len(lengths),
              "token_parity": True, "control_token_escape_parity": True, "pointer_logit_parity": True,
              "row_tokens_min": min(lengths), "row_tokens_max": max(lengths),
              "rows_over_4096": sum(n > 4096 for n in lengths),
              "head_parameters": sum(p.numel() for p in head.parameters()),
              "head_gradient_and_update": True, "head_save_reload_exact": True,
              "pretrained_backbone_loaded": False,
              "gradient_input": "random synthetic hidden states; not actual 119 features"}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
