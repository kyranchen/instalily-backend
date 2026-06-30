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
from typing import Any, Optional, TYPE_CHECKING

from .context import DISHWASHER_PREFIXES, REFRIGERATOR_PREFIXES
from .store import PartStore

if TYPE_CHECKING:
    from rag.retrieve import Retriever

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
    {
        "name": "search_parts",
        "description": (
            "Semantic search across the catalog for parts that match a free-text "
            "description of a problem, symptom, or component (e.g. 'ice maker "
            "making noise', 'dishwasher upper rack falling', 'door bin broken'). "
            "Use this when the user describes a symptom WITHOUT a specific part "
            "number. Returns up to 3 candidate parts ranked by relevance, each "
            "with the same fields as get_part_details. Returns an empty list "
            "when no part is sufficiently relevant — do NOT fabricate a result "
            "from an empty response."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Free-text description of the symptom or component the user needs",
                },
                "appliance_type": {
                    "type": "string",
                    "enum": ["Refrigerator", "Dishwasher"],
                    "description": (
                        "Optional filter to restrict results to one appliance "
                        "type. Use this whenever the user has clearly indicated "
                        "their appliance type to keep results on-topic."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_install_guide",
        "description": (
            "Get installation help for a SPECIFIC part the user has named by its "
            "PartSelect number (e.g. PS11752778) or manufacturer part number. Use "
            "this when the user asks how to install or replace a part they've "
            "identified — e.g. 'how do I install PS11752778?' or 'how do I replace "
            "this part?'. Returns install difficulty, estimated time, the "
            "installation video, and real customer installation notes. Ground any "
            "steps you give in the returned `installation_notes` — these are "
            "customer-submitted, so present them as such and do NOT invent steps. "
            "If the part isn't found, say so honestly."
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
        "name": "get_repair_guide",
        "description": (
            "Retrieve relevant prose snippets from the catalog for a given "
            "symptom or repair question (e.g. 'how to replace a dishwasher "
            "drain pump'). Returns up to 3 text excerpts plus the part numbers "
            "they came from. Use this for diagnostic or how-to questions where "
            "the user needs explanation, not just a product card. Ground your "
            "repair advice in these snippets — do not invent steps."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symptom": {
                    "type": "string",
                    "description": "What the user is trying to diagnose or repair, in plain language",
                },
                "appliance_type": {
                    "type": "string",
                    "enum": ["Refrigerator", "Dishwasher"],
                    "description": "Optional filter for appliance category",
                },
            },
            "required": ["symptom"],
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


def get_install_guide(store: PartStore, part_number: str) -> dict[str, Any]:
    """Installation help for a known part: difficulty, time, video, customer notes."""
    part = store.get(part_number)
    if not part:
        return {"found": False, "part_number": part_number.upper().strip()}
    video_id = part.get("install_video_id")
    return {
        "found": True,
        "part_number": part["part_number"],
        "name": part["name"],
        "appliance_type": part["appliance_type"],
        "install_difficulty": part.get("install_difficulty"),
        "estimated_time": part.get("install_time"),
        "install_video_youtube_id": video_id,
        "install_video_url": (
            f"https://www.youtube.com/watch?v={video_id}" if video_id else None
        ),
        # Customer-submitted DIY narratives — the closest thing to step text we have.
        "installation_notes": part.get("repair_stories", [])[:3],
        "description": part.get("description"),
        "image_url": part.get("image_url"),
        "source_url": part.get("source_url"),
    }


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
# RAG-backed tools
# ---------------------------------------------------------------------------

def search_parts(
    store: PartStore,
    retriever: "Retriever",
    query: str,
    appliance_type: Optional[str] = None,
) -> dict[str, Any]:
    hits = retriever.search(query, appliance_type=appliance_type)
    results: list[dict[str, Any]] = []
    for h in hits:
        part = store.get(h.meta["part_number"])
        if not part:
            continue
        slim = _slim_part(part)
        slim["relevance_score"] = round(h.score, 3)
        results.append(slim)
    return {
        "query": query,
        "appliance_filter": appliance_type,
        "result_count": len(results),
        "results": results,
    }


def get_repair_guide(
    retriever: "Retriever",
    symptom: str,
    appliance_type: Optional[str] = None,
) -> dict[str, Any]:
    hits = retriever.search(symptom, appliance_type=appliance_type)
    snippets: list[dict[str, Any]] = []
    for h in hits:
        snippets.append(
            {
                "part_number": h.meta["part_number"],
                "name": h.meta["name"],
                "appliance_type": h.meta["appliance_type"],
                "source_url": h.meta.get("source_url"),
                "relevance_score": round(h.score, 3),
                "snippet": h.snippet,
            }
        )
    return {
        "symptom": symptom,
        "appliance_filter": appliance_type,
        "snippet_count": len(snippets),
        "snippets": snippets,
    }


# ---------------------------------------------------------------------------
# Dispatcher used by the agent loop
# ---------------------------------------------------------------------------

TOOL_NAMES = {schema["name"] for schema in TOOL_SCHEMAS}


def run_tool(
    store: PartStore,
    retriever: "Retriever",
    name: str,
    arguments: dict[str, Any],
) -> str:
    """Run a tool by name; always returns a JSON string for the agent to consume."""
    if name == "get_part_details":
        result = get_part_details(store, arguments["part_number"])
    elif name == "get_install_guide":
        result = get_install_guide(store, arguments["part_number"])
    elif name == "check_compatibility":
        result = check_compatibility(
            store,
            arguments["part_number"],
            arguments["model_number"],
        )
    elif name == "search_parts":
        result = search_parts(
            store,
            retriever,
            arguments["query"],
            arguments.get("appliance_type"),
        )
    elif name == "get_repair_guide":
        result = get_repair_guide(
            retriever,
            arguments["symptom"],
            arguments.get("appliance_type"),
        )
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result, ensure_ascii=False)
