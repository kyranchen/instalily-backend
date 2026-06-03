"""
FastAPI entrypoint.

POST /chat       { message, session_id } -> { response, parts, tool_calls }
GET  /healthz    liveness probe

The agent reads from a JSON-backed store loaded once at startup. CORS is
configured for the CRA dev server on :3000.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(Path(__file__).parent / ".env", override=True)

from agent.context import SessionStore   # noqa: E402  (must come after load_dotenv)
from agent.loop import run_turn          # noqa: E402
from agent.store import PartStore        # noqa: E402
from rag.retrieve import Retriever       # noqa: E402

app = FastAPI(title="PartSelect Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PART_STORE = PartStore()
RETRIEVER = Retriever()
SESSIONS = SessionStore()


class ChatRequest(BaseModel):
    message: str
    session_id: str


class PartCard(BaseModel):
    part_number: str
    name: str
    price: Optional[str] = None
    image_url: Optional[str] = None
    source_url: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    parts: List[PartCard] = []
    tool_calls: List[dict] = []


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "parts_loaded": len(PART_STORE)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session = SESSIONS.get_or_create(req.session_id)
    turn = run_turn(PART_STORE, RETRIEVER, session, req.message)

    # Build product cards for any parts the agent successfully looked up this turn
    cards: list[PartCard] = []
    seen: set[str] = set()
    for ps in turn.parts_referenced:
        if ps in seen:
            continue
        seen.add(ps)
        part = PART_STORE.get(ps)
        if part:
            cards.append(
                PartCard(
                    part_number=part["part_number"],
                    name=part["name"],
                    price=part.get("price"),
                    image_url=part.get("image_url"),
                    source_url=part.get("source_url"),
                )
            )

    # Strip raw JSON from tool_calls for the wire (keep it small)
    wire_calls = [
        {"name": tc["name"], "input": tc["input"], "result": _truncate(tc["result"])}
        for tc in turn.tool_calls
    ]

    return ChatResponse(response=turn.text, parts=cards, tool_calls=wire_calls)


def _truncate(s: str, n: int = 600) -> str:
    return s if len(s) <= n else s[:n] + "...(truncated)"
