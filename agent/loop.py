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

from rag.retrieve import Retriever

from .context import Session, extract_entities
from .guardrails import (
    SAFE_FALLBACK_REPLY,
    Violation,
    build_rewrite_nudge,
    validate_turn,
)
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
    violations: list[Violation] = None  # populated only if guardrails fired


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


def run_turn(
    store: PartStore,
    retriever: Retriever,
    session: Session,
    user_message: str,
) -> TurnResult:
    client = _client()
    _update_entities_from_text(session, user_message)
    session.add_user(user_message)

    tool_calls: list[dict[str, Any]] = []
    parts_referenced: list[str] = []
    guardrail_retries = 0
    MAX_GUARDRAIL_RETRIES = 1

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
            # Final text turn — extract, then validate before returning
            text_parts = [
                b.text for b in response.content if getattr(b, "type", None) == "text"
            ]
            text = "\n".join(text_parts).strip()

            validation = validate_turn(text, tool_calls)
            if validation.ok:
                return TurnResult(
                    text=text,
                    tool_calls=tool_calls,
                    parts_referenced=parts_referenced,
                    violations=[],
                )

            if guardrail_retries < MAX_GUARDRAIL_RETRIES:
                # Inject a corrective user-role nudge and let the agent rewrite
                guardrail_retries += 1
                session.history.append(
                    {
                        "role": "user",
                        "content": build_rewrite_nudge(validation.violations),
                    }
                )
                continue

            # Out of retries: don't ship the ungrounded draft, use the safe fallback
            return TurnResult(
                text=SAFE_FALLBACK_REPLY,
                tool_calls=tool_calls,
                parts_referenced=parts_referenced,
                violations=validation.violations,
            )

        # Run every tool_use block the model emitted this round
        tool_results_block: list[dict[str, Any]] = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            tool_input = dict(block.input)
            _update_entities_from_tool_input(session, tool_input)
            result_json = run_tool(store, retriever, block.name, tool_input)
            tool_calls.append({"name": block.name, "input": tool_input, "result": result_json})
            try:
                parsed = json.loads(result_json)
                # get_part_details: top-level part_number when found
                if parsed.get("found") and parsed.get("part_number"):
                    parts_referenced.append(parsed["part_number"])
                # search_parts: each result has a part_number
                for r in parsed.get("results", []) or []:
                    if r.get("part_number"):
                        parts_referenced.append(r["part_number"])
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
