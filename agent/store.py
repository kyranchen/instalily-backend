"""
In-memory store backed by data/parts.json.

Tools call into this for deterministic lookups. Loaded once at process
start. If the underlying file changes, a server restart is required —
that's the freshness trade-off we documented for stage 1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "parts.json"


class PartStore:
    def __init__(self, path: Path = DATA_PATH) -> None:
        raw: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        self._parts: list[dict[str, Any]] = raw
        # Build indices for O(1) lookup by either ID
        self._by_ps: dict[str, dict[str, Any]] = {p["part_number"]: p for p in raw}
        self._by_mpn: dict[str, dict[str, Any]] = {
            p["manufacturer_part_number"]: p for p in raw
        }

    def get(self, identifier: str) -> dict[str, Any] | None:
        """Look up a part by PS number or manufacturer part number (case-insensitive)."""
        key = identifier.strip().upper()
        return self._by_ps.get(key) or self._by_mpn.get(key)

    def all(self) -> list[dict[str, Any]]:
        return list(self._parts)

    def __len__(self) -> int:
        return len(self._parts)
