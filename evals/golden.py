"""
Golden cases for the agent.

Twelve scenarios exercising every tool plus both guardrail rules. Each test
hits the real Anthropic API once or twice; total cost per full run is well
under $1.

To run:
    python -m pytest evals/golden.py -v

To run a single case:
    python -m pytest evals/golden.py::test_01_direct_part_lookup -v -s
"""

from __future__ import annotations

import json
import re

import pytest

from evals.runner import CaseResult, run_case, tool_inputs_for, tool_names_called

# ---------------------------------------------------------------------------
# Single-tool happy paths
# ---------------------------------------------------------------------------

def test_01_direct_part_lookup() -> None:
    """User names a specific PS number → get_part_details fires."""
    r = run_case(["Tell me about part PS11752778"])
    assert "get_part_details" in tool_names_called(r), _debug(r)
    inputs = tool_inputs_for(r, "get_part_details")
    assert any("PS11752778" in (i.get("part_number") or "").upper() for i in inputs), _debug(r)


def test_02_compatibility_clear_mismatch() -> None:
    """Fridge part + dishwasher model → check_compatibility, answer says not compatible."""
    r = run_case(["Does part PS11752778 fit my dishwasher model WDT780SAEM1?"])
    assert "check_compatibility" in tool_names_called(r), _debug(r)
    text_lower = r.text.lower()
    assert ("not compatible" in text_lower or "won't fit" in text_lower or "won’t fit" in text_lower), _debug(r)
    assert not r.violations, _debug(r)


def test_03_symptom_search_no_part_number() -> None:
    """User describes a symptom without a part number → search_parts."""
    r = run_case(["My refrigerator ice maker is making a loud grinding noise"])
    assert "search_parts" in tool_names_called(r), _debug(r)


def test_04_repair_guide_how_to() -> None:
    """How-to question → get_repair_guide."""
    r = run_case(["How do I replace the drain pump on my dishwasher?"])
    names = tool_names_called(r)
    # Either get_repair_guide or search_parts is acceptable for a how-to —
    # both ground the agent in real catalog data. Both is also fine.
    assert ("get_repair_guide" in names or "search_parts" in names), _debug(r)


# ---------------------------------------------------------------------------
# Entity memory across turns
# ---------------------------------------------------------------------------

def test_05_entity_memory_this_part() -> None:
    """Turn 1 mentions PS11752778; turn 2 says 'this part' — must resolve."""
    r = run_case(
        [
            "Tell me about PS11752778",
            "Is this part compatible with my dishwasher WDT780SAEM1?",
        ]
    )
    # Final turn must have used check_compatibility with the resolved part
    final_calls = r.turns[-1].tool_calls
    final_names = [c["name"] for c in final_calls]
    assert "check_compatibility" in final_names, _debug(r)
    compat_inputs = [c["input"] for c in final_calls if c["name"] == "check_compatibility"]
    assert any(
        "PS11752778" in (i.get("part_number") or "").upper()
        and "WDT780SAEM1" in (i.get("model_number") or "").upper()
        for i in compat_inputs
    ), _debug(r)


# ---------------------------------------------------------------------------
# Scope enforcement
# ---------------------------------------------------------------------------

def test_06_out_of_scope_refusal() -> None:
    """Washing machine question → polite refusal, zero tool calls, no API waste."""
    r = run_case(["My washing machine is making a loud grinding noise during the spin cycle"])
    assert tool_names_called(r) == [], _debug(r)
    text_lower = r.text.lower()
    assert ("refrigerator" in text_lower or "fridge" in text_lower) and "dishwasher" in text_lower, _debug(r)


# ---------------------------------------------------------------------------
# Lookups by manufacturer part number
# ---------------------------------------------------------------------------

def test_07_mpn_lookup_resolves() -> None:
    """User gives manufacturer part number, not PS — lookup still works."""
    r = run_case(["What can you tell me about part WPW10321304?"])
    assert "get_part_details" in tool_names_called(r), _debug(r)
    # The tool result should map MPN -> the corresponding PS record
    gpd_calls = [c for c in r.tool_calls if c["name"] == "get_part_details"]
    found_record = False
    for c in gpd_calls:
        try:
            parsed = json.loads(c["result"])
        except (json.JSONDecodeError, TypeError):
            continue
        if parsed.get("found") and parsed.get("manufacturer_part_number") == "WPW10321304":
            found_record = True
            break
    assert found_record, _debug(r)


# ---------------------------------------------------------------------------
# Compatibility edge cases
# ---------------------------------------------------------------------------

def test_08_likely_compatible_same_appliance() -> None:
    """Fridge part + fridge model not in sample → 'likely_compatible' status."""
    r = run_case(["Does PS11752778 fit my refrigerator WRS325SDHZ05?"])
    assert "check_compatibility" in tool_names_called(r), _debug(r)
    compat_calls = [c for c in r.tool_calls if c["name"] == "check_compatibility"]
    statuses = []
    for c in compat_calls:
        try:
            statuses.append(json.loads(c["result"]).get("status"))
        except (json.JSONDecodeError, TypeError):
            pass
    # Either confirmed-compatible (if model happens to be in sample) or
    # likely_compatible (same appliance, unconfirmed). Anything else is wrong.
    assert any(s in ("compatible", "likely_compatible") for s in statuses), f"statuses={statuses} | " + _debug(r)


def test_09_unknown_must_not_become_incompatible() -> None:
    """Nonexistent part → unknown status; reply MUST NOT say 'not compatible'.

    This is the guardrail R2 test. We accept either:
      - The agent retried after the guardrail caught it, and the final reply
        is honest about the uncertainty, OR
      - The agent never fired the violation in the first place.

    What we reject: a reply that ends with 'not compatible' / 'won't fit' /
    'incompatible' near the model number.
    """
    r = run_case(["Does part PS00000000 fit my WDT780SAEM1 dishwasher?"])
    text_lower = r.text.lower()
    has_definitive_negative = any(
        kw in text_lower
        for kw in ["not compatible", "incompatible", "won't fit", "won’t fit", "doesn't fit"]
    )
    # If the agent did make a definitive-negative claim, the guardrail should
    # have either rewritten it or fallen back to the safe reply.
    if has_definitive_negative:
        # Final text should NOT contain the violation (post-guardrail)
        assert "cannot confirm" in text_lower or "couldn't" in text_lower or "not in" in text_lower, _debug(r)


# ---------------------------------------------------------------------------
# RAG with appliance filter
# ---------------------------------------------------------------------------

def test_10_search_with_appliance_filter() -> None:
    """User clearly says 'dishwasher' — search_parts should pass appliance_type."""
    r = run_case(["I need a drain pump for my dishwasher"])
    assert "search_parts" in tool_names_called(r), _debug(r)
    sp_inputs = tool_inputs_for(r, "search_parts")
    assert any(i.get("appliance_type") == "Dishwasher" for i in sp_inputs), _debug(r)


# ---------------------------------------------------------------------------
# Multi-tool flow
# ---------------------------------------------------------------------------

def test_11_search_then_compatibility() -> None:
    """Search first, then ask compatibility about a discovered part."""
    r = run_case(
        [
            "I need a replacement door shelf bin for my Whirlpool fridge",
            "Will it work with model WRS325SDHZ05?",
        ]
    )
    # Across the conversation, both tools must have fired
    names_across = tool_names_called(r)
    assert "search_parts" in names_across, _debug(r)
    assert "check_compatibility" in names_across, _debug(r)


# ---------------------------------------------------------------------------
# Honest handling of nonexistent parts
# ---------------------------------------------------------------------------

def test_12_nonexistent_part_honest() -> None:
    """Made-up part number → reply must acknowledge the lookup failed."""
    r = run_case(["Tell me about part PS00000000"])
    assert "get_part_details" in tool_names_called(r), _debug(r)
    text_lower = r.text.lower()
    # Acceptance: any honest acknowledgement that we couldn't find it
    honest_phrases = [
        "couldn't find", "could not find", "not found", "no record",
        "doesn't appear", "does not appear", "not in our", "unable to find",
        "no information", "no result",
    ]
    assert any(p in text_lower for p in honest_phrases), _debug(r)


# ---------------------------------------------------------------------------
# Installation route (named part → get_install_guide)
# ---------------------------------------------------------------------------

def test_13_install_named_part() -> None:
    """How-to install for a NAMED part → get_install_guide (not get_part_details)."""
    r = run_case(["How do I install part PS11752778?"])
    names = tool_names_called(r)
    assert "get_install_guide" in names, _debug(r)
    # It should look up the part the user named
    gi_inputs = tool_inputs_for(r, "get_install_guide")
    assert any("PS11752778" in (i.get("part_number") or "").upper() for i in gi_inputs), _debug(r)


def test_14_install_via_entity_memory() -> None:
    """“how do I install it?” after naming a part resolves via entity memory."""
    r = run_case(
        [
            "Tell me about PS11753379",
            "How do I install it?",
        ]
    )
    final_names = [c["name"] for c in r.turns[-1].tool_calls]
    assert "get_install_guide" in final_names, _debug(r)
    gi_inputs = [c["input"] for c in r.turns[-1].tool_calls if c["name"] == "get_install_guide"]
    assert any("PS11753379" in (i.get("part_number") or "").upper() for i in gi_inputs), _debug(r)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _debug(r: CaseResult) -> str:
    """Produce a verbose failure message including tool calls and text."""
    lines = ["", "------ assistant text ------", r.text or "(empty)", "------ tool calls ------"]
    for tc in r.tool_calls:
        lines.append(f"  {tc['name']} input={tc['input']}")
    if r.violations:
        lines.append("------ violations ------")
        for v in r.violations:
            lines.append(f"  {v.rule}: {v.detail}")
    return "\n".join(lines)
