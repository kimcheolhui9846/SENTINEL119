"""Character n-gram evidence for full-call Zoro dispatch and urgency decisions."""

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from .prepare import QUESTIONS


LABELS = {name: list(question["criteria"]) for name, question in QUESTIONS.items()}
FEATURE_COUNT = 2 ** 19


def counts(states):
    vectorizer = HashingVectorizer(analyzer="char", ngram_range=(2, 5), n_features=FEATURE_COUNT,
                                  alternate_sign=False, norm=None, dtype=np.float32)
    return vectorizer.transform(states)


def tfidf(matrix, idf):
    matrix.data = 1 + np.log(matrix.data)
    matrix.data *= idf[matrix.indices]
    return normalize(matrix, copy=False)


def lexical_scores(states, weights):
    matrix = tfidf(counts(states), weights["idf"])
    scores = {}
    for name in LABELS:
        margins = matrix @ weights[name + "_coef"].T + weights[name + "_intercept"]
        # Independent one-vs-rest probabilities, on a common log scale for blending.
        scores[name] = -np.logaddexp(0, -margins)
    return scores


def combine_scores(name, pointer_scores, transcript_scores, policy):
    """Input arrays use LABELS order. The urgency offset is in log-score units."""
    blend = policy["transcript_weight"]
    score = (1 - blend) * pointer_scores + blend * transcript_scores
    if name == "urgency":
        direction = {"high": 1, "medium": 0, "low": -1}
        score = score + policy["severity_offset"] * np.array([direction[k] for k in LABELS[name]])
    return score


def predict_full_call(state, model, tokenizer, weights, policy, max_length=4096, prompt=None):
    """Apply a selected policy to one transcript using its matching Zoro checkpoint.

    ``prompt`` is the checkpoint's recorded ``run.json["prompt"]``; omit it for bare-prompt checkpoints.
    """
    import torch
    from .decision import encode
    from .prompts import apply as apply_prompt

    if policy["identity"]["labels"] != LABELS:
        raise ValueError("Unexpected class order in full-call policy")
    rows = encode(apply_prompt({"state": state, "questions": QUESTIONS}, prompt), tokenizer, max_length)
    text = lexical_scores([state], weights)
    model.eval()
    result = {}
    with torch.no_grad():
        for row, logits in zip(rows, model(rows), strict=True):
            name = row["qid"]
            by_key = dict(zip(row["keys"], logits.float().tolist(), strict=True))
            pointer = np.array([by_key[k] for k in LABELS[name]])
            scores = combine_scores(name, pointer, text[name][0], policy["tasks"][name])
            result[name] = LABELS[name][int(scores.argmax())]
    return result
