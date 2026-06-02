"""
Structured (deterministic) tools.

Two tools in this layer — both read-only, both backed by the local store:
  get_part_details   : pull a part record by PS# or MPN
  check_compatibility: decide whether a part fits a specific appliance model

Both tools return JSON-serializable dicts. They never raise on "not found";
they return a `found: false` / status: "unknown" payload so the agent (and
the stage-3 guardrails) can reason about absence honestly.
"""

from __future__ import annotations

import json
from typing import Any

from .context import DISHWASHER_PREFIXES, REFRIGERATOR_PREFIXES
from .store import PartStore

# ---------------------------------------------------------------------------
# Tool schemas — match Anthropic's tool-use input_schema format
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_part_details",
        "description": (
            "Look up a specific PartSelect part by its PartSelect number "
            "(e.g. PS11752778) or by manufacturer part number (e.g. WPW10321304). "
            "Returns price, name, description, symptoms it fixes, brand, install "
            "difficulty, video, and a sample of compatible models. Use this any "
            "time the user asks about a specific part by ID."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "part_number": {
                    "type": "string",
                    "description": "PartSelect number (PS prefix) or manufacturer part number",
                }
            },
            "required": ["part_number"],
        },
    },
    {
        "name": "check_compatibility",
        "description": (
            "Check whether a specific part fits a specific appliance model. "
            "Returns one of: 'compatible' (confirmed match), 'likely_compatible' "
            "(appliance types match but exact model match unconfirmed), "
            "'not_compatible' (appliance types do not match), or 'unknown' (we "
            "lack enough data to decide). You MUST call this tool before making "
            "any compatibility claim — never assert compatibility from prior "
            "knowledge or by reading product names."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "part_number": {
                    "type": "string",
                    "description": "PartSelect number or manufacturer part number for the part",
                },
                "model_number": {
                    "type": "string",
                    "description": "Appliance model number, e.g. WDT780SAEM1",
                },
            },
            "required": ["part_number", "model_number"],
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _appliance_from_model(model_number: str) -> str | None:
    m = model_number.upper().strip()
    if any(m.startswith(p) for p in DISHWASHER_PREFIXES):
        return "Dishwasher"
    if any(m.startswith(p) for p in REFRIGERATOR_PREFIXES):
        return "Refrigerator"
    return None


def _slim_part(part: dict[str, Any]) -> dict[str, Any]:
    """Return a compact, agent-friendly view of a part."""
    return {
        "found": True,
        "part_number": part["part_number"],
        "manufacturer_part_number": part["manufacturer_part_number"],
        "name": part["name"],
        "appliance_type": part["appliance_type"],
        "price": part.get("price"),
        "in_stock": part.get("in_stock"),
        "brand": part.get("brand"),
        "brands_fits": part.get("brands_fits", []),
        "description": part.get("description"),
        "symptoms_fixed": part.get("symptoms", []),
        "replaces_part_numbers": part.get("replaces_part_numbers", []),
        "sample_compatible_models": part.get("sample_models", []),
        "install_difficulty": part.get("install_difficulty"),
        "install_video_youtube_id": part.get("install_video_id"),
        "image_url": part.get("image_url"),
        "source_url": part.get("source_url"),
    }


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def get_part_details(store: PartStore, part_number: str) -> dict[str, Any]:
    part = store.get(part_number)
    if not part:
        return {"found": False, "part_number": part_number.upper().strip()}
    return _slim_part(part)


def check_compatibility(
    store: PartStore, part_number: str, model_number: str
) -> dict[str, Any]:
    part = store.get(part_number)
    model_clean = model_number.upper().strip()
    part_clean = part_number.upper().strip()

    if not part:
        return {
            "status": "unknown",
            "part_number": part_clean,
            "model_number": model_clean,
            "reason": "Part not found in catalog; cannot determine compatibility.",
        }

    sample_models: list[str] = [m.upper() for m in part.get("sample_models", [])]
    part_appliance = part.get("appliance_type")
    model_appliance = _appliance_from_model(model_clean)

    # Strongest signal: model appears in the confirmed compatibility sample
    if model_clean in sample_models:
        return {
            "status": "compatible",
            "part_number": part_clean,
            "model_number": model_clean,
            "reason": f"Model {model_clean} is in this part's confirmed compatibility list.",
        }

    # Appliance type mismatch is a definitive negative
    if model_appliance and part_appliance and part_appliance != "Unknown":
        if model_appliance != part_appliance:
            return {
                "status": "not_compatible",
                "part_number": part_clean,
                "model_number": model_clean,
                "reason": (
                    f"This part is for a {part_appliance}; model {model_clean} "
                    f"appears to be a {model_appliance}."
                ),
            }
        # Same appliance type, model not in sample — likely compatible but unconfirmed
        return {
            "status": "likely_compatible",
            "part_number": part_clean,
            "model_number": model_clean,
            "reason": (
                f"Both the part and model are for {part_appliance}s, but model "
                f"{model_clean} is not in our cached sample of confirmed-compatible "
                f"models. Verify on the part's source page before purchase."
            ),
            "sample_compatible_models": sample_models[:5],
        }

    # Couldn't classify the model from its prefix; be honest
    return {
        "status": "unknown",
        "part_number": part_clean,
        "model_number": model_clean,
        "reason": (
            f"Model {model_clean} did not match any known appliance prefix, so "
            f"we can't confirm or rule out compatibility from cached data."
        ),
        "sample_compatible_models": sample_models[:5],
    }


# ---------------------------------------------------------------------------
# Dispatcher used by the agent loop
# ---------------------------------------------------------------------------

TOOL_NAMES = {schema["name"] for schema in TOOL_SCHEMAS}


def run_tool(store: PartStore, name: str, arguments: dict[str, Any]) -> str:
    """Run a tool by name; always returns a JSON string for the agent to consume."""
    if name == "get_part_details":
        result = get_part_details(store, arguments["part_number"])
    elif name == "check_compatibility":
        result = check_compatibility(
            store,
            arguments["part_number"],
            arguments["model_number"],
        )
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result, ensure_ascii=False)
