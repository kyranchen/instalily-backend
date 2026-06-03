"""
Eval harness runner.

Spins up a fresh session per case (or per case + turn list) and invokes the
real agent loop end-to-end. Returns a structured TurnResult so the eval can
assert on tool calls, text content, and guardrail violations.

We deliberately call the live Anthropic API rather than mocking — the whole
point of the eval is to verify that the model selects the right tool given
our prompts and tool descriptions. Mocking the LLM would only test our
dispatcher (already covered by unit tests).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env before importing the agent (which checks for the API key).
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from agent.context import Session                # noqa: E402
from agent.loop import TurnResult, run_turn      # noqa: E402
from agent.store import PartStore                # noqa: E402
from rag.retrieve import Retriever               # noqa: E402


# Module-level singletons — load once across all tests so we don't pay the
# embedder/store load cost per case.
_STORE: PartStore | None = None
_RETRIEVER: Retriever | None = None


def _store() -> PartStore:
    global _STORE
    if _STORE is None:
        _STORE = PartStore()
    return _STORE


def _retriever() -> Retriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever()
    return _RETRIEVER


@dataclass
class CaseResult:
    text: str                          # last assistant text
    tool_calls: list[dict[str, Any]]   # union across all turns
    violations: list[Any]              # guardrail violations from last turn
    turns: list[TurnResult]            # per-turn results, in order
    session: Session


def run_case(messages: list[str], session_id: str = "eval") -> CaseResult:
    """Run a single- or multi-turn case in a fresh session."""
    assert messages, "must pass at least one user message"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set — paste it into .env")

    session = Session(session_id=session_id)
    all_calls: list[dict[str, Any]] = []
    turns: list[TurnResult] = []
    for msg in messages:
        tr = run_turn(_store(), _retriever(), session, msg)
        turns.append(tr)
        all_calls.extend(tr.tool_calls)

    last = turns[-1]
    return CaseResult(
        text=last.text,
        tool_calls=all_calls,
        violations=last.violations or [],
        turns=turns,
        session=session,
    )


def tool_names_called(result: CaseResult) -> list[str]:
    return [tc["name"] for tc in result.tool_calls]


def tool_inputs_for(result: CaseResult, name: str) -> list[dict[str, Any]]:
    return [tc["input"] for tc in result.tool_calls if tc["name"] == name]
