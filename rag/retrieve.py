"""
RAG retrieval.

Loads the precomputed embeddings + metadata at construction time and serves
similarity searches. The embedder model is loaded too, since query strings
need to be embedded at request time.

Two parameters worth caring about:

  threshold (default 0.25)
      Cosine similarity below this is treated as "no relevant match" and
      omitted. Tunes precision vs. recall — higher means stricter.

  top_k (default 3)
      Maximum results to return after threshold filtering.

The metadata filter (appliance_type) is applied BEFORE top-k, so an
appliance filter that excludes everything yields an empty result, not a
weak best-of-irrelevant result. That's the right behavior for a guardrail
("we don't have any dishwasher parts that match").
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VECTORS_PATH = DATA / "vectors.npy"
META_PATH = DATA / "vectors_meta.json"
DOCS_DIR = DATA / "docs"

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.25
DEFAULT_TOP_K = 3


@dataclass
class Hit:
    score: float
    meta: dict[str, Any]
    snippet: str  # ~500-char excerpt from the doc, useful for repair-guide tool


class Retriever:
    def __init__(
        self,
        vectors_path: Path = VECTORS_PATH,
        meta_path: Path = META_PATH,
        model_name: str = MODEL_NAME,
    ) -> None:
        if not vectors_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"RAG index missing. Run `python -m rag.embed` first. "
                f"Looked for {vectors_path.name} and {meta_path.name}."
            )
        self._vectors: np.ndarray = np.load(vectors_path)
        self._meta: list[dict[str, Any]] = json.loads(meta_path.read_text())
        assert self._vectors.shape[0] == len(self._meta), "index/meta length mismatch"
        self._model = SentenceTransformer(model_name)

    def search(
        self,
        query: str,
        appliance_type: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> list[Hit]:
        if not query.strip():
            return []

        # Apply metadata filter first — keeps irrelevant categories from being
        # returned even when nothing in-scope is a good match.
        candidate_indices: list[int]
        if appliance_type:
            filt = appliance_type.strip().lower()
            candidate_indices = [
                i
                for i, m in enumerate(self._meta)
                if (m.get("appliance_type") or "").lower() == filt
            ]
        else:
            candidate_indices = list(range(len(self._meta)))

        if not candidate_indices:
            return []

        q_vec = self._model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0].astype(np.float32)
        # Cosine sim = dot product since both sides are L2-normalized
        candidate_matrix = self._vectors[candidate_indices]
        scores = candidate_matrix @ q_vec  # shape (len(candidate_indices),)

        # Pair (idx, score), sort desc, take top_k that clear threshold
        ranked = sorted(
            zip(candidate_indices, scores), key=lambda t: t[1], reverse=True
        )

        hits: list[Hit] = []
        for idx, score in ranked[:top_k]:
            if score < threshold:
                break
            meta = self._meta[idx]
            snippet = _read_snippet(meta.get("doc_path", ""))
            hits.append(Hit(score=float(score), meta=meta, snippet=snippet))
        return hits


def _read_snippet(doc_path_rel: str, max_chars: int = 800) -> str:
    """Return a short excerpt from a doc file (used by get_repair_guide)."""
    if not doc_path_rel:
        return ""
    p = ROOT / doc_path_rel
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8")
    return text[:max_chars] + ("..." if len(text) > max_chars else "")
