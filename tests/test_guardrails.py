"""
Unit tests for agent/guardrails.py — verify both rules fire on the
violations they're supposed to catch, and don't fire on benign drafts.
"""

from __future__ import annotations

import json

from agent.guardrails import (
    NEGATIVE_VERDICT_RE,
    POSITIVE_VERDICT_RE,
    validate_turn,
)


def _compat_call(part: str, model: str, status: str) -> dict:
    return {
        "name": "check_compatibility",
        "input": {"part_number": part, "model_number": model},
        "result": json.dumps({"status": status, "model_number": model}),
    }


# ---------------------------------------------------------------------------
# R1: ungrounded compatibility claims
# ---------------------------------------------------------------------------

def test_r1_fires_when_verdict_about_model_has_no_tool_call() -> None:
    text = "Yes, PS11752778 is compatible with your WDT780SAEM1 dishwasher."
    r = validate_turn(text, tool_calls=[])
    assert not r.ok
    assert any(v.rule == "UNGROUNDED_COMPAT_CLAIM" for v in r.violations)
    assert any(v.model_number == "WDT780SAEM1" for v in r.violations)


def test_r1_quiet_when_tool_was_called_for_that_model() -> None:
    text = "Yes, PS11752778 is compatible with your WDT780SAEM1 dishwasher."
    r = validate_turn(text, [_compat_call("PS11752778", "WDT780SAEM1", "compatible")])
    assert r.ok, r.violations


def test_r1_quiet_when_no_model_number_mentioned() -> None:
    # Generic statements without a specific model are fine.
    text = "This part fits many side-by-side refrigerators from Whirlpool and Kenmore."
    r = validate_turn(text, tool_calls=[])
    assert r.ok, r.violations


def test_r1_quiet_when_model_mentioned_without_verdict() -> None:
    # Naming a model in passing isn't a claim.
    text = "If you tell me your model number — say, WDT780SAEM1 — I can check."
    r = validate_turn(text, tool_calls=[])
    assert r.ok, r.violations


def test_r1_fires_for_negative_verdict_too() -> None:
    text = "PS11752778 won't fit your WDT780SAEM1; it's a refrigerator part."
    r = validate_turn(text, tool_calls=[])
    assert not r.ok
    assert any(v.rule == "UNGROUNDED_COMPAT_CLAIM" for v in r.violations)


def test_r1_fires_for_each_distinct_ungrounded_model() -> None:
    # Two different models, neither backed by a tool call → two violations.
    text = (
        "Yes, this part is compatible with WDT780SAEM1, and it also fits WRS325SDHZ05."
    )
    r = validate_turn(text, tool_calls=[])
    models_flagged = {v.model_number for v in r.violations if v.rule == "UNGROUNDED_COMPAT_CLAIM"}
    assert models_flagged == {"WDT780SAEM1", "WRS325SDHZ05"}


# ---------------------------------------------------------------------------
# R2: unknown -> incompatible escalation
# ---------------------------------------------------------------------------

def test_r2_fires_when_unknown_becomes_not_compatible() -> None:
    text = "Sorry, PS11752778 is not compatible with WDT780SAEM1."
    calls = [_compat_call("PS11752778", "WDT780SAEM1", "unknown")]
    r = validate_turn(text, calls)
    assert not r.ok
    assert any(v.rule == "UNKNOWN_AS_INCOMPATIBLE" for v in r.violations)


def test_r2_quiet_when_unknown_kept_as_uncertain() -> None:
    text = (
        "I couldn't confirm compatibility for PS11752778 with WDT780SAEM1 from "
        "the catalog. Please verify on the PartSelect product page."
    )
    calls = [_compat_call("PS11752778", "WDT780SAEM1", "unknown")]
    r = validate_turn(text, calls)
    assert r.ok, r.violations


def test_r2_quiet_when_status_is_actually_not_compatible() -> None:
    # If the tool genuinely returned not_compatible, a negative verdict is fine.
    text = "PS11752778 is not compatible with WDT780SAEM1 — different appliance types."
    calls = [_compat_call("PS11752778", "WDT780SAEM1", "not_compatible")]
    r = validate_turn(text, calls)
    assert r.ok, r.violations


# ---------------------------------------------------------------------------
# Regex sanity — keep the patterns from drifting
# ---------------------------------------------------------------------------

def test_negative_verdict_patterns_match() -> None:
    samples = [
        "not compatible",
        "incompatible with your model",
        "won't fit your dishwasher",
        "won’t fit",       # curly apostrophe
        "will not fit",
        "doesn't fit",
        "does not work",
        "is not compatible",
        "isn't compatible",
    ]
    for s in samples:
        assert NEGATIVE_VERDICT_RE.search(s), f"negative pattern missed: {s!r}"


def test_positive_verdict_patterns_match() -> None:
    samples = [
        "is compatible",
        "are compatible",
        "compatible with your dishwasher",
        "will fit your model",
        "fits your model",
        "works with your appliance",
    ]
    for s in samples:
        assert POSITIVE_VERDICT_RE.search(s), f"positive pattern missed: {s!r}"


def test_verdict_patterns_dont_match_benign_text() -> None:
    benign = [
        "this product description",
        "it is the right size",
        "please share the model number",
        "fits many appliances",  # generic, no model — handled by R1 proximity gate
    ]
    for s in benign:
        # We only care that the COMBINED gate doesn't fire; individual patterns
        # may match harmlessly because the proximity-to-model check is the real gate.
        r = validate_turn(s, tool_calls=[])
        assert r.ok, f"benign sentence flagged: {s!r} -> {r.violations}"
