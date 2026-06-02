"""
Session + entity memory.

Two responsibilities:
  1. Track conversation history per session_id so the agent has continuity.
  2. Resolve pronouns/anaphora ("this part", "my model") into concrete IDs by
     remembering the most recently mentioned part and model.

Entity extraction runs on every user message and every tool call so that by
the time the agent drafts a reply, current_part / current_model reflect the
freshest reference. The agent sees these surfaced inline in the system prompt
each turn.

History is capped to MAX_TURNS most-recent message pairs to bound token use.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any

MAX_TURNS = 12  # last 12 user/assistant turn-pairs

# Whirlpool/KitchenAid/etc. dishwasher and refrigerator model prefixes.
# Used both for extraction and (in tools.py) for appliance-type inference.
DISHWASHER_PREFIXES = ("WDT", "WDF", "WDP", "KDPE", "KDFE", "KDTE", "MDB", "GDF", "GDT")
REFRIGERATOR_PREFIXES = (
    "WRS", "WRX", "WRF", "WRT", "WRB", "KRFC", "KRFF", "KRMF", "KSFB",
    "MFI", "MFF", "MSC", "FFSS", "GSS", "10640",  # Kenmore numeric prefix
)

PART_NUMBER_RE = re.compile(r"\b(PS\d{4,9})\b", re.IGNORECASE)
# Manufacturer parts: alphanumeric, 6+ chars, contains at least one digit, may
# have leading letters (W, WP, AP). Restrictive enough to avoid false positives
# on common words.
MPN_RE = re.compile(r"\b((?:WP?|AP|EDR)\w*\d{4,}\w*)\b")
MODEL_RE = re.compile(
    r"\b("
    + "|".join(DISHWASHER_PREFIXES + REFRIGERATOR_PREFIXES)
    + r")[A-Z0-9]{2,12}\b",
    re.IGNORECASE,
)


@dataclass
class Session:
    session_id: str
    history: list[dict[str, Any]] = field(default_factory=list)
    current_part: str | None = None        # PS number when known, else MPN
    current_model: str | None = None

    def add_user(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: Any) -> None:
        # `content` may be a string or a list of content blocks (Anthropic format)
        self.history.append({"role": "assistant", "content": content})
        self._trim()

    def add_tool_result(self, tool_use_id: str, result: Any) -> None:
        """Tool results live as a user-role message with tool_result blocks."""
        block = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result if isinstance(result, str) else str(result),
                }
            ],
        }
        self.history.append(block)
        self._trim()

    def _trim(self) -> None:
        # Keep last MAX_TURNS*2 entries (one per role). Always start with a user message.
        max_entries = MAX_TURNS * 2
        if len(self.history) > max_entries:
            # Drop from the front but never split a tool_use / tool_result pair —
            # walk forward until we land on a `user` text message.
            self.history = self.history[-max_entries:]
            while self.history and not _is_clean_start(self.history[0]):
                self.history.pop(0)


def _is_clean_start(msg: dict[str, Any]) -> bool:
    """A clean conversation start is a plain user text message."""
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if isinstance(content, str):
        return True
    # If it's a list, it must not contain a tool_result (those need a preceding tool_use)
    if isinstance(content, list):
        return not any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        )
    return False


def extract_entities(text: str) -> tuple[str | None, str | None]:
    """Return (part_id, model_id) detected in the text, or (None, None)."""
    part: str | None = None
    if m := PART_NUMBER_RE.search(text):
        part = m.group(1).upper()
    elif m := MPN_RE.search(text):
        # Tighter check: must contain a digit and at least 6 chars
        cand = m.group(1).upper()
        if any(c.isdigit() for c in cand) and len(cand) >= 6:
            part = cand

    model: str | None = None
    if m := MODEL_RE.search(text):
        model = m.group(0).upper()
    return part, model


class SessionStore:
    """Thread-safe in-memory session registry. Single-process only."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> Session:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = Session(session_id=session_id)
            return self._sessions[session_id]
