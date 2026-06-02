"""
Claude tool-use loop.

Single function `run_turn` that takes a user message + session and returns
the assistant's final text reply. Internally it iterates with Anthropic's
Messages API: while the model emits `tool_use` blocks, run the tools, feed
results back, and continue. Stops on `end_turn` or a hard iteration cap.

Side effects on the session:
  - user message and final assistant text appended to history
  - intermediate tool_use / tool_result rounds appended (so the next turn
    can reason about what the agent already looked up)
  - entity extraction runs on the user message and on every tool call, so
    current_part / current_model stay fresh.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic

from .context import Session, extract_entities
from .prompts import build_system_prompt
from .store import PartStore
from .tools import TOOL_SCHEMAS, run_tool

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024
MAX_TOOL_ROUNDS = 6


@dataclass
class TurnResult:
    text: str                          # final assistant message text
    tool_calls: list[dict[str, Any]]   # list of {name, input, result} for this turn
    parts_referenced: list[str]        # PS numbers the agent looked up successfully


def _client() -> Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return Anthropic(api_key=key)


def _update_entities_from_text(session: Session, text: str) -> None:
    p, m = extract_entities(text)
    if p:
        session.current_part = p
    if m:
        session.current_model = m


def _update_entities_from_tool_input(session: Session, tool_input: dict[str, Any]) -> None:
    if pn := tool_input.get("part_number"):
        p, _ = extract_entities(str(pn))
        if p:
            session.current_part = p
        else:
            session.current_part = str(pn).upper().strip()
    if mn := tool_input.get("model_number"):
        _, m = extract_entities(str(mn))
        if m:
            session.current_model = m
        else:
            session.current_model = str(mn).upper().strip()


def run_turn(store: PartStore, session: Session, user_message: str) -> TurnResult:
    client = _client()
    _update_entities_from_text(session, user_message)
    session.add_user(user_message)

    tool_calls: list[dict[str, Any]] = []
    parts_referenced: list[str] = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=build_system_prompt(session.current_part, session.current_model),
            tools=TOOL_SCHEMAS,
            messages=session.history,
        )

        # Append the assistant turn (full content blocks, since tool_use rounds
        # require the original blocks in subsequent calls)
        session.add_assistant(response.content)

        if response.stop_reason != "tool_use":
            # Final text turn — extract text blocks
            text_parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
            return TurnResult(
                text="\n".join(text_parts).strip(),
                tool_calls=tool_calls,
                parts_referenced=parts_referenced,
            )

        # Run every tool_use block the model emitted this round
        tool_results_block: list[dict[str, Any]] = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            tool_input = dict(block.input)
            _update_entities_from_tool_input(session, tool_input)
            result_json = run_tool(store, block.name, tool_input)
            tool_calls.append({"name": block.name, "input": tool_input, "result": result_json})
            try:
                parsed = json.loads(result_json)
                if parsed.get("found") and parsed.get("part_number"):
                    parts_referenced.append(parsed["part_number"])
            except json.JSONDecodeError:
                pass
            tool_results_block.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_json,
                }
            )

        # All tool_results for this round must be in one user message
        session.history.append({"role": "user", "content": tool_results_block})

    # Hit the iteration cap without a final text — degrade gracefully
    return TurnResult(
        text=(
            "I wasn't able to fully resolve that lookup. Could you rephrase or "
            "share the specific part or model number you're asking about?"
        ),
        tool_calls=tool_calls,
        parts_referenced=parts_referenced,
    )
