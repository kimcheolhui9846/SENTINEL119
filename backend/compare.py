"""Model comparison tab: one call, one transcript prefix at a time, to the plain LLM, SENTINEL_ZORO and base Zoro.

Independent of the dispatch session. SENTINEL_ZORO_URL points at the SENTINEL_ZORO model server
(/v1/stream, /v1/llm). SENTINEL_SCENARIO_DIRS (os.pathsep-separated) holds call files
{id, title, summary, turns: [{speaker: 0 dispatcher | 1 caller, text}], gold?}. Files under a "private"
directory are real AI Hub calls for local demos only; keep them out of the repository.
"""

import json
import os
from pathlib import Path
from typing import Annotated, Literal

import httpx

from .call_area import find_area
from .demo_harness import apply_harness, load_profile, parse_json
from .inference import InferenceError
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field


class CompareTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speaker: Literal[0, 1]
    text: Annotated[str, Field(min_length=1, max_length=2000)]


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    turns: Annotated[list[CompareTurn], Field(min_length=1, max_length=120)]


def _calls():
    """Comparison calls come from the same cached, validated loader as the session scenarios."""
    from .scenarios import _call_files
    return {key: {"id": key, "title": c["title"], "summary": c["summary"], "private": c["private"], "gold": c["gold"],
                  "area": c["area"], "turns": [{"speaker": 0 if who == "수보요원" else 1, "text": text}
                                               for who, text in c["turns"]]}
            for key, c in _call_files().items()}


def _calls_uncached():
    calls = {}
    for root in filter(None, os.getenv("SENTINEL_SCENARIO_DIRS", "").split(os.pathsep)):
        for path in sorted(Path(root).glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            calls[data["id"]] = {"id": data["id"], "title": data["title"], "summary": data.get("summary", ""),
                                 "private": "private" in path.parts, "gold": data.get("gold"),
                                 "area": find_area(data.get("address"), seed=data["id"]),
                                 "turns": [{"speaker": 1 if t["speaker"] else 0, "text": t["text"]} for t in data["turns"]]}
    return calls


def _server():
    raw = os.getenv("SENTINEL_ZORO_URL", "").strip()
    try:
        url = httpx.URL(raw) if raw else None
    except httpx.InvalidURL:
        url = None
    if url is None or url.scheme not in {"http", "https"} or not url.host or url.userinfo or url.query:
        raise HTTPException(503, "compare_unconfigured")
    return str(url).rstrip("/")


def _relay(path, body, timeout):
    try:
        response = httpx.post(_server() + path, json=body, timeout=timeout, trust_env=False)
    except httpx.HTTPError:
        raise HTTPException(502, "compare_unavailable") from None
    if response.status_code != 200:
        raise HTTPException(502, "compare_unavailable")
    try:
        value = parse_json(response.content)
        # The original payload is relayed, including metadata the harness does not read.
        json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, RecursionError):
        raise HTTPException(502, "compare_unavailable") from None
    return value


router = APIRouter(prefix="/api/compare")


@router.get("")
def compare_calls():
    return {"scenarios": [{**{key: call[key] for key in ("id", "title", "summary", "private")},
                           "area": f"{call['area']['district']} {call['area']['dong']}" if call["area"] else None}
                          for call in _calls().values()]}


@router.get("/{scenario_id}")
def compare_call(scenario_id: str):
    call = _calls().get(scenario_id)
    if call is None:
        raise HTTPException(404, "scenario_not_found")
    return call


@router.post("/decide")
def compare_decide(body: CompareRequest):
    try:
        profile = load_profile()
    except ValueError:
        raise HTTPException(503, "compare_unconfigured") from None
    turns = body.model_dump()["turns"]
    response = _relay("/v1/stream", {"turns": turns, "variants": ["sft", "base"]}, 10.0)
    try:
        harness = apply_harness(turns, response, profile)
    except InferenceError:
        raise HTTPException(502, "compare_unavailable") from None
    return {**response, "harness": harness}


@router.post("/llm")
def compare_llm(body: CompareRequest):
    return _relay("/v1/llm", {"turns": body.model_dump()["turns"]}, 30.0)
