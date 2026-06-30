"""
One-off embedding script.

Reads every file in data/docs/, embeds it with all-MiniLM-L6-v2, and writes
two artifacts the retriever loads at server start:

  data/vectors.npy          float32 [N, 384] L2-normalized matrix
  data/vectors_meta.json    list[N] of per-doc metadata (part_number,
                            appliance_type, doc_path, source_url, etc.)

Order is preserved: row i in vectors.npy corresponds to entry i in the meta
list. Re-run this whenever data/parts.json or data/docs/ change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOCS = DATA / "docs"
PARTS_JSON = DATA / "parts.json"
VECTORS = DATA / "vectors.npy"
META = DATA / "vectors_meta.json"

MODEL_NAME = "all-MiniLM-L6-v2"
SNIPPET_CHARS = 800  # excerpt length stored in the index for get_repair_guide


def main() -> None:
    if not PARTS_JSON.exists():
        print("data/parts.json not found — run scrape.py first", file=sys.stderr)
        sys.exit(1)

    parts = {p["part_number"]: p for p in json.loads(PARTS_JSON.read_text())}

    doc_files = sorted(DOCS.glob("*.txt"))
    if not doc_files:
        print("no docs in data/docs/ — run scrape.py first", file=sys.stderr)
        sys.exit(1)

    print(f"[embed] {len(doc_files)} docs, model={MODEL_NAME}", file=sys.stderr)
    model = SentenceTransformer(MODEL_NAME)

    texts: list[str] = []
    meta: list[dict] = []
    for path in doc_files:
        ps = path.stem  # "PS11752778"
        part = parts.get(ps)
        if not part:
            print(f"  skip {ps} (not in parts.json)", file=sys.stderr)
            continue
        text = path.read_text(encoding="utf-8")
        texts.append(text)
        meta.append(
            {
                "part_number": part["part_number"],
                "manufacturer_part_number": part["manufacturer_part_number"],
                "name": part["name"],
                "appliance_type": part["appliance_type"],
                "symptoms": part.get("symptoms", []),
                "source_url": part.get("source_url"),
                "image_url": part.get("image_url"),
                "price": part.get("price"),
                "doc_path": str(path.relative_to(ROOT)),
                # Bake the snippet into the index so retrieval needs no doc file
                # at request time — data/docs/ is then a pure embedding input.
                "snippet": text[:SNIPPET_CHARS] + ("..." if len(text) > SNIPPET_CHARS else ""),
            }
        )

    embeddings = model.encode(
        texts,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize so cosine sim = dot product
        show_progress_bar=False,
    ).astype(np.float32)

    np.save(VECTORS, embeddings)
    META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[embed] wrote {embeddings.shape[0]} vectors "
        f"({embeddings.shape[1]} dim) to {VECTORS.name}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
