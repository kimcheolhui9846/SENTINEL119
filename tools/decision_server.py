"""Zoro-1 decision server for the SENTINEL119 demo.

Runs in the GPU training environment, next to the ``sentinel119`` package, so that the call
rendering, question text and a checkpoint's recorded prompt are exactly those used in training.

  POST /v1/classify  dispatch + urgency for the current turns (variant "base" or "sft")
  POST /v1/choose    one choice question over demo candidates (vehicle, hospital); base head
  POST /v1/stream    turn-wise dispatch / subtype / urgency (+ symptom) for sft and base on one prefix
  POST /v1/llm       plain-LLM baseline (--llm, e.g. Qwen/Qwen3.5-4B): generated JSON answer for the same prefix
  GET  /v1/health    pinned model identity, available variants and checkpoint hashes
  GET  /v1/models     console-compatible model list: "sentinel-zoro" (sft) and "zoro-base"
  POST /v1/systemone  console-compatible typed choices. dispatch/urgency with the console's Korean labels
                      run the trained question text and keys (sft adds hierarchy answers); any other choice
                      question runs the released head over the given state

Vehicle and hospital choices always use the released head with the adapter disabled: the SFT
adapter was trained and evaluated only on dispatch/urgency (docs/spec.md §6).

Run:  python tools/decision_server.py --checkpoint runs/zoro-hier-h1b [--llm Qwen/Qwen3.5-4B] --port 8119
      (pip install -e ".[train,serve]"; the server binds 127.0.0.1 by default, reach it over an SSH tunnel)
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import threading
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch

from sentinel119.decision import ContextOverflow, encode
from sentinel119.model import load_head, load_zoro
from sentinel119.prepare import QUESTIONS

MAX_LENGTH = 4096


class Turn(BaseModel):
    speaker: int = Field(ge=0, le=1, description="0 = 수보자, 1 = 신고자 (AI Hub code)")
    text: str = Field(min_length=1)


class ClassifyRequest(BaseModel):
    turns: list[Turn] = Field(min_length=1)
    variant: str = "sft"


class StreamRequest(BaseModel):
    turns: list[Turn] = Field(min_length=1)
    variants: list[str] = ["sft", "base"]


class LlmRequest(BaseModel):
    turns: list[Turn] = Field(min_length=1)


LLM_PROMPT = """다음은 119 신고 통화의 지금까지 대화입니다.
{call}

지금까지의 대화만 보고 아래 JSON 한 줄로만 답하세요. 근거가 아직 없으면 null을 쓰세요.
- 출동유형: 화재, 구조, 구급, 기타 중 하나
- 세부유형: 출동유형이 구급이면 {ems} 중 하나, 구조이면 {rescue} 중 하나, 그 밖에는 null
- 긴급도: 상, 중, 하 중 하나
{{"출동유형": ..., "세부유형": ..., "긴급도": ...}}"""


class Generator:
    """Plain LLM baseline: the base Qwen generates a JSON answer (reasoning off) for the same prefix."""

    def __init__(self, model_id, device="cuda"):
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
        self.lock = threading.Lock()

    def answer(self, turns, labels):
        call = "\n".join(f"{'수보요원' if t.speaker == 0 else '신고자'}: {t.text}" for t in turns)
        prompt = LLM_PROMPT.format(call=call, ems=", ".join(labels["ems_subtype"].values()),
                                   rescue=", ".join(labels["rescue_subtype"].values()))
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        ids = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self.lock, torch.no_grad():
            start = time.perf_counter()
            out = self.model.generate(**ids, max_new_tokens=80, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
        raw = self.tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return raw, elapsed, int(out.shape[1] - ids["input_ids"].shape[1])


def parse_llm(raw, labels):
    """Map the generated Korean labels back to keys; anything outside the option set becomes invalid."""
    import re
    reverse = {q: {v: k for k, v in opts.items()} for q, opts in labels.items()}
    match = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    def pick(value, qid):
        if value in (None, "null", ""):
            return None, False
        key = reverse.get(qid, {}).get(str(value).strip())
        return key, key is None
    dispatch, bad_d = pick(data.get("출동유형"), "dispatch")
    sub_q = {"ems": "ems_subtype", "rescue": "rescue_subtype"}.get(dispatch)
    subtype, bad_s = pick(data.get("세부유형"), sub_q) if sub_q else (None, False)
    urgency, bad_u = pick(data.get("긴급도"), "urgency")
    return {"dispatch": dispatch, "subtype": subtype, "subtype_question": sub_q, "urgency": urgency,
            "invalid": [n for n, bad in (("출동유형", bad_d), ("세부유형", bad_s), ("긴급도", bad_u)) if bad] or ([] if data else ["형식"])}


class ChoiceQuestion(BaseModel):
    instructions: str
    criteria: dict[str, str] = Field(min_length=1)


class ChooseRequest(BaseModel):
    turns: list[Turn] = Field(min_length=1)
    question: ChoiceQuestion


def render_state(turns):
    """Identical to sentinel119.prepare.make_example."""
    return "\n".join(f"화자{t.speaker}: {t.text}" for t in turns)


class Decider:
    def __init__(self, checkpoint=None, device="cuda", local_files_only=True):
        self.lock = threading.Lock()
        prior = None
        kwargs = {}
        self.checkpoint = None
        if checkpoint:
            prior = json.loads((checkpoint / "run.json").read_text(encoding="utf-8"))
            if prior.get("status") != "completed":
                raise ValueError("Checkpoint did not complete")
            kwargs["head_path"] = checkpoint / "decision_head.pt"
            if prior["mode"] == "lora":
                kwargs["adapter_path"] = checkpoint / "adapter"
            files = [kwargs["head_path"]]
            if "adapter_path" in kwargs:
                files += [kwargs["adapter_path"] / n for n in ("adapter_config.json", "adapter_model.safetensors")]
            self.checkpoint = {"path": checkpoint.name, "mode": prior["mode"],
                               "max_length": prior["config"]["max_length"],
                               "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
        self.model, self.tokenizer, self.identity = load_zoro(device=device, local_files_only=local_files_only, **kwargs)
        self.model.eval()
        self.heads = {"base": load_head(Path(self.identity["local_model_dir"]) / "decision_head.pt", device)}
        if checkpoint:
            self.heads["sft"] = self.model.head
        self.prior = prior
        self.prompt = (prior or {}).get("prompt", {"variant": "bare", "overrides": {}})
        self.max_length = self.checkpoint["max_length"] if self.checkpoint else MAX_LENGTH

    def _scores(self, example, variant):
        if variant not in self.heads:
            raise HTTPException(422, f"Variant {variant} is not loaded; available: {sorted(self.heads)}")
        try:
            rows = encode(example, self.tokenizer, self.max_length)
        except ContextOverflow as exc:
            raise HTTPException(413, str(exc)) from None
        backbone = self.model.backbone
        adapter_off = variant == "base" and hasattr(backbone, "disable_adapter")
        with self.lock, torch.no_grad(), (backbone.disable_adapter() if adapter_off else nullcontext()):
            start = time.perf_counter()
            logits = self.heads[variant](*self.model.features(rows))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
        results, offset = {}, 0
        for row in rows:
            z = logits[offset:offset + len(row["opts"])].float()
            offset += len(row["opts"])
            results[row["qid"]] = {"probabilities": dict(zip(row["keys"], torch.softmax(z, -1).tolist())),
                                   "logits": dict(zip(row["keys"], z.tolist()))}
        return results, elapsed

    def stream_profile(self):
        """Which question set the loaded checkpoint was trained with."""
        manifest = (self.prior or {}).get("dataset_manifest", {})
        if manifest.get("hierarchy"):
            return "hierarchy"
        return "auxiliary" if manifest.get("auxiliary") else "basic"

    def stream_questions(self):
        """Turn-wise question set with hierarchy keys; returns (questions, rename for auxiliary keys)."""
        from sentinel119.hierarchy import EMS_SUBTYPES, QUESTIONS as HIERARCHY, RESCUE_SUBTYPES
        questions = {"dispatch": QUESTIONS["dispatch"], "urgency": QUESTIONS["urgency"]}
        if self.stream_profile() != "auxiliary":
            questions.update(HIERARCHY)
            return questions, {}
        # Auxiliary checkpoints learned one 16-way subtype question; ask it restricted to one parent's children.
        from sentinel119.auxiliary import AUX_QUESTIONS, SUBTYPES
        sub = AUX_QUESTIONS["subtype"]
        rename = {}
        for (large, medium), key in SUBTYPES.items():
            if large == "구급":
                rename[key] = ("ems_subtype", EMS_SUBTYPES[medium])
            elif large == "구조":
                rename[key] = ("rescue_subtype", RESCUE_SUBTYPES[medium])
        for qid in ("ems_subtype", "rescue_subtype"):
            questions[qid] = {**sub, "criteria": {k: v for k, v in sub["criteria"].items()
                                                  if rename.get(k, (None,))[0] == qid}}
        return questions, rename

    def stream(self, turns, variant):
        from sentinel119.prompts import apply as apply_prompt
        questions, rename = self.stream_questions()
        example = {"state": render_state(turns), "questions": questions}
        if variant == "sft":
            example = apply_prompt(example, self.prompt)
        results, elapsed = self._scores(example, variant)
        out = {}
        for qid, result in results.items():
            probs = result["probabilities"]
            if rename and qid in ("ems_subtype", "rescue_subtype"):
                probs = {rename[k][1]: p for k, p in probs.items()}
            out[qid] = probs
        return out, elapsed

    def describe(self, variant):
        return {"model_id": self.identity["model_id"], "revision": self.identity["revision"],
                "variant": variant, "checkpoint": self.checkpoint if variant == "sft" else None,
                "prompt_variant": self.prompt["variant"] if variant == "sft" else "bare"}


def stream_labels():
    from sentinel119.hierarchy import QUESTIONS as HIERARCHY
    return {"dispatch": dict(QUESTIONS["dispatch"]["criteria"]), "urgency": dict(QUESTIONS["urgency"]["criteria"]),
            **{qid: dict(q["criteria"]) for qid, q in HIERARCHY.items()}}


CONSOLE_MODELS = {"sentinel-zoro": "sft", "zoro-base": "base"}
CONSOLE_SPEAKERS = {"수보요원": 0, "수보자": 0, "dispatcher": 0}
GENERIC_INSTRUCTIONS = "현재 상태를 보고 가장 적절한 선택지를 고르세요."


class SystemOneRequest(BaseModel):
    model: str
    state: dict | str
    questions: dict[str, dict]
    reasoning: bool = False


def console_turns(state):
    """Console turns {speaker: "신고자"|"수보요원", text} -> training turns (0 = 수보자, 1 = 신고자)."""
    if not isinstance(state, dict) or not isinstance(state.get("turns"), list):
        return None
    turns = [Turn(speaker=CONSOLE_SPEAKERS.get(t.get("speaker"), 1), text=t["text"])
             for t in state["turns"] if isinstance(t, dict) and isinstance(t.get("text"), str) and t["text"].strip()]
    return turns or None


def systemone(decider, request):
    variant = CONSOLE_MODELS.get(request.model)
    if variant is None or variant not in decider.heads:
        raise HTTPException(404, f"Unknown model {request.model}; available: {sorted(CONSOLE_MODELS)}")
    labels = stream_labels()
    turns = console_turns(request.state)
    answers, hierarchy, elapsed = {}, {}, 0.0
    trained = {qid for qid, q in request.questions.items() if qid in ("dispatch", "urgency")
               and set((q.get("criteria") or {})) == set(labels[qid].values())}
    if trained:
        if turns is None:
            raise HTTPException(422, "dispatch/urgency need state.turns")
        probs, ms = decider.stream(turns, variant)
        elapsed += ms
        for qid in trained:
            dist = {labels[qid][key]: p for key, p in probs[qid].items()}
            answers[qid] = {"type": "choice", "choice": max(dist, key=dist.get), "probabilities": dist}
        for qid in ("ems_subtype", "rescue_subtype", "ems_symptom"):
            if qid in probs:
                hierarchy[qid] = {labels[qid].get(k, k): p for k, p in probs[qid].items()}
    rest = {qid: q for qid, q in request.questions.items() if qid not in trained}
    if rest:
        state = render_state(turns) if turns else (
            request.state if isinstance(request.state, str) else json.dumps(request.state, ensure_ascii=False))
        questions = {qid: {"type": "choice", "instructions": q.get("instructions") or GENERIC_INSTRUCTIONS,
                           "criteria": {k: (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
                                        for k, v in (q.get("criteria") or {}).items()}}
                     for qid, q in rest.items()}
        if any(q.get("type", "choice") != "choice" or not q.get("criteria") for q in rest.values()):
            raise HTTPException(422, "Only nonempty choice questions are supported")
        results, ms = decider._scores({"state": state, "questions": questions}, variant)
        elapsed += ms
        for qid, result in results.items():
            dist = result["probabilities"]
            answers[qid] = {"type": "choice", "choice": max(dist, key=dist.get), "probabilities": dist}
    for answer in answers.values():  # float64 renormalization: consoles check the sum strictly
        total = sum(answer["probabilities"].values())
        answer["probabilities"] = {k: v / total for k, v in answer["probabilities"].items()}
    return {"model": request.model, "answers": answers, "hierarchy": hierarchy, "latency_ms": elapsed}


def create_app(decider, generator=None):
    app = FastAPI(title="SENTINEL119 Zoro decision server")

    @app.get("/v1/health")
    def health():
        return {"status": "ok", "variants": sorted(decider.heads), "identity": decider.identity,
                "checkpoint": decider.checkpoint, "prompt_variant": decider.prompt["variant"]}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": name, "object": "model", "variant": variant}
                                           for name, variant in CONSOLE_MODELS.items() if variant in decider.heads]}

    @app.post("/v1/systemone")
    def system_one(request: SystemOneRequest):
        return systemone(decider, request)

    @app.post("/v1/classify")
    def classify(request: ClassifyRequest):
        from sentinel119.prompts import apply as apply_prompt
        example = {"state": render_state(request.turns), "questions": QUESTIONS}
        if request.variant == "sft":
            example = apply_prompt(example, decider.prompt)
        results, elapsed = decider._scores(example, request.variant)
        return {"model": decider.describe(request.variant), "results": results, "latency_ms": elapsed}

    @app.post("/v1/stream")
    def stream(request: StreamRequest):
        """Turn-wise decisions for every requested variant on the same transcript prefix."""
        labels = stream_labels()
        out = {}
        for variant in request.variants:
            probs, elapsed = decider.stream(request.turns, variant)
            out[variant] = {"model": decider.describe(variant), "probabilities": probs, "latency_ms": elapsed}
        return {"profile": decider.stream_profile(), "labels": labels, "results": out,
                "state_chars": len(render_state(request.turns))}

    @app.post("/v1/llm")
    def llm(request: LlmRequest):
        if generator is None:
            raise HTTPException(503, "LLM baseline is not loaded; start the server with --llm")
        labels = stream_labels()
        raw, elapsed, new_tokens = generator.answer(request.turns, labels)
        return {"model": {"model_id": generator.model_id, "variant": "llm"}, "text": raw, "answer": parse_llm(raw, labels),
                "latency_ms": elapsed, "new_tokens": new_tokens}

    @app.post("/v1/choose")
    def choose(request: ChooseRequest):
        question = {"type": "choice", "instructions": request.question.instructions,
                    "criteria": request.question.criteria}
        results, elapsed = decider._scores({"state": render_state(request.turns),
                                            "questions": {"choice": question}}, "base")
        return {"model": decider.describe("base"), "results": results["choice"], "latency_ms": elapsed}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, help="Completed sentinel119 training output; omit for Base only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8119)
    parser.add_argument("--allow-download", action="store_true", help="Permit fetching the pinned revision")
    parser.add_argument("--llm", help="Also load a plain LLM baseline for /v1/llm, e.g. Qwen/Qwen3.5-4B")
    args = parser.parse_args()
    import uvicorn
    decider = Decider(args.checkpoint, args.device, local_files_only=not args.allow_download)
    generator = Generator(args.llm, args.device) if args.llm else None
    uvicorn.run(create_app(decider, generator), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
