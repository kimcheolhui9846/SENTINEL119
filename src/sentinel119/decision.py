"""Text-only choice encoding compatible with Zoro's direct decision protocol.

The delimiter/escaping protocol follows trida-stack and Kev (Apache-2.0);
see THIRD_PARTY_NOTICES.md. Unlike upstream, overlong states are rejected,
never truncated underneath a full-call label.
"""

import json
import re

SPECIAL = ("<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>")
CONTROL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
MODEL_ID = "trillionlabs/Zoro-1"
MODEL_REVISION = "bf1b76b9b4df03b05fcbb73444e1cd4db0f566da"
UPSTREAM_REVISION = "0ddcf516e458a82892d7fe0d0dabeb113d3cf424"
MAX_STATE, MAX_BRANCH = 7552, 8192


class ContextOverflow(ValueError):
    pass


def user_tokens(tokenizer, text):
    return tokenizer(CONTROL.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def encode(example, tokenizer, max_length=4096, rng=None):
    """Each question gets its own causal row; labels/meta never enter the token stream."""
    if not isinstance(example.get("state"), str) or not example["state"].strip():
        raise ValueError("Expected a nonempty text state")
    if not example.get("questions"):
        raise ValueError("Expected choice questions")
    vocab = tokenizer.get_vocab()
    if any(token not in vocab for token in SPECIAL):
        raise ValueError("Tokenizer lacks Zoro decision delimiter tokens")
    s, q, o, end, decide = (tokenizer.convert_tokens_to_ids(t) for t in SPECIAL)
    state = [s] + user_tokens(tokenizer, example["state"])
    if len(state) > MAX_STATE:
        raise ContextOverflow(f"Full-call state needs {len(state)} tokens; Zoro limit is {MAX_STATE}")
    rows = []
    for qid, question in example["questions"].items():
        criteria = question.get("criteria")
        if question.get("type") != "choice" or not isinstance(criteria, dict) or not criteria:
            raise ValueError("Only nonempty choice questions are supported")
        keys = list(criteria)
        if rng is not None:
            rng.shuffle(keys)
        label = question.get("label")
        if label is not None and label not in keys:
            raise ValueError("Label is outside the allowed choices")
        ids = state + [q] + user_tokens(tokenizer, question.get("instructions", ""))
        ends = []
        for key in keys:
            description = criteria[key]
            if not isinstance(description, str):
                raise ValueError("Choice descriptions must be strings")
            option = f"{key}: {description}" if description else key
            ids.extend([o] + user_tokens(tokenizer, option) + [end])
            ends.append(len(ids) - 1)
        ids.append(decide)
        if len(ids) > min(max_length, MAX_BRANCH):
            raise ContextOverflow(f"Question {qid} needs {len(ids)} tokens; limit is {max_length}")
        rows.append({"ids": ids, "decide": len(ids) - 1, "opts": ends, "qid": qid,
                     "keys": keys, "label": keys.index(label) if label is not None else None})
    return rows


def read_examples(path, expected_hash=None):
    import hashlib
    payload = path.read_bytes()
    if expected_hash and hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError(f"Dataset hash mismatch: {path.name}")
    # Records end at "\n" only; str.splitlines() would also split at U+2028/U+2029 inside JSON strings.
    examples = [json.loads(line) for line in payload.decode("utf-8").split("\n") if line.strip()]
    if not examples:
        raise ValueError(f"Empty dataset: {path.name}")
    for example in examples:
        meta = example.get("_meta", {})
        if meta.get("label_scope") != "full_call" or not meta.get("group_id"):
            raise ValueError("Only prepared full-call examples with group IDs are supported")
        if any("label" not in question for question in example["questions"].values()):
            raise ValueError("Missing supervised label")
    return examples
