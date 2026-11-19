import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from sentinel119.transcript import LABELS, combine_scores, lexical_scores, predict_full_call
from test_training import Tokenizer


def test_full_call_matches_offline_policy_across_key_orders(monkeypatch):
    torch = pytest.importorskip("torch")
    from sentinel119.transcript import FEATURE_COUNT
    from sentinel119.prepare import QUESTIONS

    state = "화자1: 합성 통화입니다."
    weights = {"idf": np.ones(FEATURE_COUNT, dtype=np.float32)}
    pointer = {"urgency": [-1., .2, .25], "dispatch": [0., 1., 2., .5]}
    tasks = {}
    for name, labels in LABELS.items():
        criteria = QUESTIONS[name]["criteria"]
        monkeypatch.setitem(QUESTIONS[name], "criteria", dict(reversed(list(criteria.items()))))
        weights[name + "_coef"] = np.zeros((len(labels), FEATURE_COUNT), dtype=np.float32)
        weights[name + "_intercept"] = np.arange(len(labels), dtype=np.float32)
        tasks[name] = {"transcript_weight": .5, "severity_offset": .5}

    class Model(torch.nn.Module):
        def forward(self, rows):
            assert all(row["label"] is None for row in rows)
            return [torch.tensor([pointer[r["qid"]][LABELS[r["qid"]].index(k)] for k in r["keys"]])
                    for r in rows]

    text = lexical_scores([state], weights)
    expected = {name: labels[combine_scores(name, np.array(pointer[name]), text[name][0], tasks[name]).argmax()]
                for name, labels in LABELS.items()}
    assert expected == {"urgency": "medium", "dispatch": "ems"}
    assert predict_full_call(state, Model(), Tokenizer(), weights,
                             {"identity": {"labels": LABELS}, "tasks": tasks}) == expected
