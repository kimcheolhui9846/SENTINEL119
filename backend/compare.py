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
    calls = {}
    for root in filter(None, os.getenv("SENTINEL_SCENARIO_DIRS", "").split(os.pathsep)):
        for path in sorted(Path(root).glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            calls[data["id"]] = {"id": data["id"], "title": data["title"], "summary": data.get("summary", ""),
                                 "private": "private" in path.parts, "gold": data.get("gold"),
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
    return response.json()


router = APIRouter(prefix="/api/compare")


@router.get("")
def compare_calls():
    return {"scenarios": [{key: call[key] for key in ("id", "title", "summary", "private")}
                          for call in _calls().values()]}


@router.get("/{scenario_id}")
def compare_call(scenario_id: str):
    call = _calls().get(scenario_id)
    if call is None:
        raise HTTPException(404, "scenario_not_found")
    return call


@router.post("/decide")
def compare_decide(body: CompareRequest):
    return _relay("/v1/stream", {"turns": body.model_dump()["turns"], "variants": ["sft", "base"]}, 10.0)


@router.post("/llm")
def compare_llm(body: CompareRequest):
    return _relay("/v1/llm", {"turns": body.model_dump()["turns"]}, 30.0)
