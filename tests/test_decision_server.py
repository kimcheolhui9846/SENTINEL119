"""Decision server contract: console-compatible endpoints, label mapping and the LLM answer parser."""
import importlib.util
from pathlib import Path
import sys
import unittest

try:
    import fastapi  # noqa: F401
    from fastapi.testclient import TestClient
    import torch  # noqa: F401
except ImportError:  # serve/train extras not installed
    TestClient = None

ROOT = Path(__file__).resolve().parents[1]


def load_server():
    spec = importlib.util.spec_from_file_location("decision_server", ROOT / "tools" / "decision_server.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["decision_server"] = module
    spec.loader.exec_module(module)
    return module


class FakeDecider:
    heads = {"base": None, "sft": None}
    prompt = {"variant": "bare"}
    checkpoint = {"path": "zoro-hier-h1b", "mode": "lora"}
    identity = {"model_id": "trillionlabs/Zoro-1"}

    def __init__(self):
        self.calls = []

    def stream(self, turns, variant):
        self.calls.append(("stream", [(t.speaker, t.text) for t in turns], variant))
        return {"dispatch": {"fire": 0.1, "rescue": 0.1, "ems": 0.7, "other": 0.1},
                "urgency": {"high": 0.6, "medium": 0.3, "low": 0.1},
                "ems_subtype": {"cardiac_arrest": 0.9, "illness": 0.1}}, 5.0

    def _scores(self, example, variant):
        self.calls.append(("scores", example, variant))
        return {qid: {"probabilities": {k: 1 / len(q["criteria"]) for k in q["criteria"]}}
                for qid, q in example["questions"].items()}, 3.0

    def stream_profile(self):
        return "hierarchy"

    def describe(self, variant):
        return {"variant": variant}


@unittest.skipIf(TestClient is None, "needs the serve and train extras")
class DecisionServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = load_server()

    def setUp(self):
        self.decider = FakeDecider()
        self.client = TestClient(self.server.create_app(self.decider))

    def test_models_lists_loaded_variants(self):
        ids = [m["id"] for m in self.client.get("/v1/models").json()["data"]]
        self.assertEqual(ids, ["sentinel-zoro", "zoro-base"])

    def test_console_intake_uses_trained_questions_and_korean_labels(self):
        body = {"model": "sentinel-zoro",
                "state": {"turns": [{"id": 1, "speaker": "수보요원", "text": "119입니다"},
                                    {"id": 2, "speaker": "신고자", "text": "쓰러졌어요"}]},
                "questions": {"dispatch": {"type": "choice", "criteria": {k: k for k in ("화재", "구조", "구급", "기타")}},
                              "urgency": {"type": "choice", "criteria": {k: k for k in ("상", "중", "하")}}}}
        result = self.client.post("/v1/systemone", json=body).json()
        self.assertEqual(self.decider.calls[0], ("stream", [(0, "119입니다"), (1, "쓰러졌어요")], "sft"))
        self.assertEqual(result["answers"]["dispatch"]["choice"], "구급")
        self.assertEqual(result["answers"]["urgency"]["choice"], "상")
        self.assertAlmostEqual(sum(result["answers"]["dispatch"]["probabilities"].values()), 1.0, places=12)
        self.assertEqual(max(result["hierarchy"]["ems_subtype"], key=result["hierarchy"]["ems_subtype"].get), "심정지")

    def test_other_choices_use_the_given_state_and_criteria(self):
        body = {"model": "zoro-base", "state": {"incident_type": "구급", "candidates": [{"id": "A"}]},
                "questions": {"vehicle": {"type": "choice", "criteria": {"A": "구급차 A", "B": "구급차 B"}}}}
        result = self.client.post("/v1/systemone", json=body).json()
        kind, example, variant = self.decider.calls[0]
        self.assertEqual((kind, variant), ("scores", "base"))
        self.assertIn("incident_type", example["state"])
        self.assertEqual(set(result["answers"]["vehicle"]["probabilities"]), {"A", "B"})

    def test_unknown_model_is_rejected(self):
        body = {"model": "other", "state": "x", "questions": {"q": {"type": "choice", "criteria": {"a": "a"}}}}
        self.assertEqual(self.client.post("/v1/systemone", json=body).status_code, 404)

    def test_malformed_choice_question_is_a_client_error(self):
        body = {"model": "zoro-base", "state": "x", "questions": {"vehicle": {"type": "choice", "criteria": ["A", "B"]}}}
        self.assertEqual(self.client.post("/v1/systemone", json=body).status_code, 422)

    def test_hierarchy_questions_follow_what_the_checkpoint_learned(self):
        class Stub:
            def __init__(self, prior):
                self.prior = prior

            def stream_profile(self):
                return self.server.Decider.stream_profile(self)

        Stub.server = self.server
        ask = self.server.Decider.stream_questions
        basic = Stub({"dataset_manifest": {}, "config": {}})
        self.assertEqual(set(ask(basic)[0]), {"dispatch", "urgency"})
        subset = Stub({"dataset_manifest": {"hierarchy": True}, "config": {"questions": "dispatch,urgency,ems_subtype"}})
        self.assertEqual(set(ask(subset)[0]), {"dispatch", "urgency", "ems_subtype"})
        full = Stub({"dataset_manifest": {"hierarchy": True}, "config": {}})
        self.assertEqual(set(ask(full)[0]), {"dispatch", "urgency", "ems_subtype", "rescue_subtype", "ems_symptom"})
        self.assertEqual(len(ask(Stub(None))[0]), 5)

    def test_llm_parser_maps_labels_and_flags_invalid_values(self):
        labels = self.server.stream_labels()
        answer = self.server.parse_llm('답: {"출동유형": "구급", "세부유형": "심정지", "긴급도": "최상"}', labels)
        self.assertEqual((answer["dispatch"], answer["subtype"], answer["urgency"]), ("ems", "cardiac_arrest", None))
        self.assertEqual(answer["invalid"], ["긴급도"])
        self.assertEqual(self.server.parse_llm("모르겠습니다", labels)["invalid"], ["형식"])


if __name__ == "__main__":
    unittest.main()
