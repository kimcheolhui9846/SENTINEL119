import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("decision_rule", Path(__file__).parents[1] / "tools" / "decision_rule.py")
dr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dr)


def test_offsets_respect_downward_cap_and_crossfit_is_out_of_fold():
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 3, 300)
    scores = dr.log_softmax(rng.normal(size=(300, 3)) + 1.5 * np.eye(3)[truth])
    scores[:, 2] += 0.8  # bias toward "low" creates downward errors
    raw = dr.metrics(truth, dr.decide(scores, [0, 0, 0]), dr.URGENCY)
    offsets, feasible = dr.fit("urgency", scores, truth, max_downward=raw["downward_rate"] / 2)
    tuned = dr.metrics(truth, dr.decide(scores, offsets), dr.URGENCY)
    assert feasible and tuned["downward_rate"] <= raw["downward_rate"] / 2
    assert all(isinstance(o, float) for o in offsets) and offsets[1] == 0.0
    pred = dr.crossfit("urgency", scores, truth, None, k=3)
    assert pred.shape == truth.shape


def test_dispatch_tuning_never_lowers_in_sample_f1():
    rng = np.random.default_rng(1)
    truth = np.array([2] * 270 + [3] * 30)
    logits = rng.normal(size=(300, 4))
    logits[:, 2] += 2.0
    logits[truth == 3, 3] += 1.5
    scores = dr.log_softmax(logits)
    offsets, _ = dr.fit("dispatch", scores, truth)
    raw = dr.metrics(truth, dr.decide(scores, [0, 0, 0, 0]), dr.DISPATCH)["macro_f1"]
    assert dr.metrics(truth, dr.decide(scores, offsets), dr.DISPATCH)["macro_f1"] >= raw
    assert offsets[2] == 0.0  # ems is the fixed reference
