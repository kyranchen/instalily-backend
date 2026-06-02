"""
Post-draft validation layer.

The agent loop produces a final text reply plus a trace of the tools it
called. Before we hand the reply back to the user, we check two invariants:

  R1. UNGROUNDED_COMPAT_CLAIM
      If the reply asserts a verdict about a specific appliance model
      (e.g. "PS... is compatible with WDT780SAEM1"), then
      `check_compatibility` MUST have been called this turn with that
      model_number. Otherwise the agent is making up the answer.

  R2. UNKNOWN_AS_INCOMPATIBLE
      If `check_compatibility` returned status="unknown" for a model,
      the reply must NOT state a definitive negative verdict about that
      same model. "Not in our data" must never become "not compatible".

Both rules are conservative — false positives cost an LLM round-trip, so
the patterns are tight. We only flag when a model number AND a verdict
word both appear; verdict words alone (e.g. "this part fits many
side-by-side refrigerators") don't trigger.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .context import MODEL_RE

# Definitive negative verdicts — "I cannot confirm" / "we don't have data"
# style phrasing is excluded because it is the correct hedge for `unknown`.
NEGATIVE_VERDICT_RE = re.compile(
    r"\b(?:"
    r"not\s+compatible|"
    r"incompatible|"
    r"won['’]?t\s+(?:fit|work|be\s+compatible)|"
    r"will\s+not\s+(?:fit|work)|"
    r"doesn['’]?t\s+(?:fit|work)|"
    r"does\s+not\s+(?:fit|work)|"
    r"isn['’]?t\s+compatible|"
    r"is\s+not\s+compatible"
    r")\b",
    re.IGNORECASE,
)

# Positive verdicts — affirmative claims of fit/compatibility.
POSITIVE_VERDICT_RE = re.compile(
    r"\b(?:"
    r"is\s+compatible|"
    r"are\s+compatible|"
    r"compatible\s+with|"
    r"will\s+fit|"
    r"does\s+fit|"
    r"fits\s+(?:your|the|model)|"
    r"works\s+with\s+(?:your|model)"
    r")\b",
    re.IGNORECASE,
)

VERDICT_RE = re.compile(
    f"(?:{NEGATIVE_VERDICT_RE.pattern})|(?:{POSITIVE_VERDICT_RE.pattern})",
    re.IGNORECASE,
)

# Window (chars) around a model number where we look for verdict language.
PROXIMITY_WINDOW = 120


@dataclass
class Violation:
    rule: str          # "UNGROUNDED_COMPAT_CLAIM" | "UNKNOWN_AS_INCOMPATIBLE"
    detail: str        # human-readable, used in the rewrite nudge
    model_number: str  # the model the violation centers on


@dataclass
class ValidationResult:
    ok: bool
    violations: list[Violation] = field(default_factory=list)


def _models_in_text(text: str) -> list[tuple[str, int]]:
    """Return (model_number, start_index) for every model match in text."""
    return [(m.group(0).upper(), m.start()) for m in MODEL_RE.finditer(text)]


def _has_verdict_near(text: str, idx: int, window: int = PROXIMITY_WINDOW) -> bool:
    """Is there verdict language within `window` chars on either side of idx?"""
    start = max(0, idx - window)
    end = min(len(text), idx + window)
    return bool(VERDICT_RE.search(text[start:end]))


def _has_negative_near(text: str, idx: int, window: int = PROXIMITY_WINDOW) -> bool:
    start = max(0, idx - window)
    end = min(len(text), idx + window)
    return bool(NEGATIVE_VERDICT_RE.search(text[start:end]))


def _compat_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter and parse the check_compatibility calls from this turn."""
    out: list[dict[str, Any]] = []
    for tc in tool_calls:
        if tc.get("name") != "check_compatibility":
            continue
        try:
            parsed_result = json.loads(tc["result"]) if isinstance(tc["result"], str) else tc["result"]
        except json.JSONDecodeError:
            parsed_result = {}
        out.append(
            {
                "model_number": str(tc.get("input", {}).get("model_number", "")).upper(),
                "part_number": str(tc.get("input", {}).get("part_number", "")).upper(),
                "status": parsed_result.get("status"),
            }
        )
    return out


def validate_turn(response_text: str, tool_calls: list[dict[str, Any]]) -> ValidationResult:
    """Run R1 and R2 against the draft. Returns a ValidationResult."""
    violations: list[Violation] = []
    text = response_text or ""

    compat_calls = _compat_calls(tool_calls)
    grounded_models = {c["model_number"] for c in compat_calls if c["model_number"]}
    unknown_models = {
        c["model_number"]
        for c in compat_calls
        if c["status"] == "unknown" and c["model_number"]
    }

    seen_models_in_text = _models_in_text(text)

    # R1: every verdict claim about a model needs a backing tool call
    flagged_for_r1: set[str] = set()
    for model, idx in seen_models_in_text:
        if model in flagged_for_r1:
            continue
        if not _has_verdict_near(text, idx):
            continue
        if model not in grounded_models:
            flagged_for_r1.add(model)
            violations.append(
                Violation(
                    rule="UNGROUNDED_COMPAT_CLAIM",
                    detail=(
                        f"Your reply makes a compatibility verdict about model "
                        f"{model} but `check_compatibility` was never called for "
                        f"that model this turn. Either call the tool or remove "
                        f"the verdict and explain you'd need to check."
                    ),
                    model_number=model,
                )
            )

    # R2: if the tool returned unknown, the reply must not assert a definitive negative
    for model, idx in seen_models_in_text:
        if model not in unknown_models:
            continue
        if _has_negative_near(text, idx):
            violations.append(
                Violation(
                    rule="UNKNOWN_AS_INCOMPATIBLE",
                    detail=(
                        f"`check_compatibility` returned status `unknown` for "
                        f"model {model}, but your reply asserts a definitive "
                        f"negative verdict near that model. Rephrase to reflect "
                        f"uncertainty: we couldn't confirm compatibility, not "
                        f"that it's incompatible."
                    ),
                    model_number=model,
                )
            )

    return ValidationResult(ok=not violations, violations=violations)


def build_rewrite_nudge(violations: list[Violation]) -> str:
    """Compose a corrective system note to nudge the agent into a rewrite."""
    lines = [
        "Your previous draft failed an internal validation check. Specifically:",
    ]
    for v in violations:
        lines.append(f"- [{v.rule}] {v.detail}")
    lines.append(
        "Please rewrite your answer. If you need to call a tool to support a "
        "claim, do so now; otherwise revise the wording to be honest about "
        "what we do and do not know."
    )
    return "\n".join(lines)


SAFE_FALLBACK_REPLY = (
    "I want to give you an accurate answer, but I wasn't able to verify the "
    "compatibility claim against our catalog. Could you share the exact "
    "PartSelect number (PS-prefix) and the appliance model number, and I'll "
    "check them directly?"
)
