# PartSelect Agent — Backend

A tool-calling customer-support agent for PartSelect's refrigerator and
dishwasher parts catalog. The LLM acts as a **router**: it decides which tool
to call, and the tools — not the model — are the source of truth for part
details, compatibility, and search. Three reliability layers wrap the agent so
it stays accurate, in-scope, and honest about what it does and doesn't know.

Backend is FastAPI (Python) exposing `POST /chat`. The companion
[Create React App frontend](https://github.com/kyranchen/instalily-case-study)
calls it.

---

## Why this design

The hard problem in a parts-support bot isn't generating fluent text — it's
**not lying about compatibility**. "Does this part fit my model?" has a correct
answer that lives in data, and a plausible-sounding wrong answer the model will
happily produce from pattern-matching on product names. The whole architecture
is built around making the wrong answer structurally difficult to emit.

Three ideas do most of the work:

1. **Tools are the source of truth, the LLM is a router.** The model never
   states a price, a compatibility verdict, or a part spec from its own
   knowledge. It calls a tool, and the tool reads from a local store. Structured
   facts (price, compatibility) come from deterministic lookups; fuzzy matching
   (symptom → part) comes from RAG with a relevance threshold.

2. **A guardrail validates the draft before the user sees it.** After the agent
   writes its reply, a validation layer checks two invariants: any compatibility
   verdict must trace to a `check_compatibility` call this turn, and a "we don't
   know" tool result must never become a confident "incompatible." Violations
   trigger a rewrite; a second failure falls back to a safe response.

3. **An offline eval harness pins the behavior.** Twelve golden cases assert the
   right tool fires for the right prompt, so prompt or tool changes can't
   silently regress tool selection.

---

## Architecture

```
                  POST /chat { message, session_id }
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │  1. Session / Context Manager (context.py)   │
        │     • per-session conversation history       │
        │     • entity memory: "this part" / "my model"│
        │       → resolved to concrete IDs via regex   │
        └─────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │  2. Agent Loop (loop.py)                     │
        │     Claude tool-use loop, ≤6 tool rounds.    │
        │     Tools:                                   │
        │       • get_part_details      (structured)   │
        │       • check_compatibility   (structured)   │
        │       • search_parts          (RAG)          │
        │       • get_repair_guide      (RAG)          │
        └─────────────────────────────────────────────┘
                              │  draft reply + tool trace
                              ▼
        ┌─────────────────────────────────────────────┐
        │  3. Guardrail / Validation (guardrails.py)   │
        │     • compat claim ⇒ tool was called         │
        │     • "unknown" ≠ "not compatible"           │
        │     • on violation: 1 rewrite, then fallback │
        └─────────────────────────────────────────────┘
                              │
                              ▼
              { response, parts[], tool_calls[] }

   Sources of truth:  data/parts.json   (structured lookups)
                      data/vectors.npy  (RAG, embedded from data/docs/)
```

A separate **offline eval harness** (`evals/`) exercises the whole loop against
golden cases — it is not part of the request path.

---

## Tool inventory

| Tool | Type | When the agent uses it | Backed by |
|------|------|------------------------|-----------|
| `get_part_details` | Structured | User names a specific part (PS# or MPN) | `parts.json` |
| `check_compatibility` | Structured | "Does part X fit model Y?" | `parts.json` |
| `search_parts` | RAG | User describes a symptom, no part number | `vectors.npy` |
| `get_repair_guide` | RAG | How-to / diagnostic question | `vectors.npy` |

`check_compatibility` returns one of four statuses — `compatible`,
`likely_compatible`, `not_compatible`, `unknown` — so the agent (and the
guardrail) can distinguish "confirmed fit" from "same appliance type but
unconfirmed" from "we have no data." That four-way split is what lets the
guardrail enforce "unknown never becomes incompatible."

All four tools are **read-only**. There is no cart or order mutation — the bot
is scoped to customer support over catalog knowledge.

---

## Setup

Requires Python 3.9+ and an Anthropic API key.

> **Apple Silicon note:** install into a virtual environment. The native deps
> (`numpy`, `sentence-transformers`) ship per-architecture wheels, and mixing a
> system `pip --user` install across two Python builds causes `mach-o ...
> incompatible architecture` errors. A venv pins one interpreter and sidesteps
> this entirely.

```bash
cd instalily-backend

# 1. Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Dependencies
pip install -r requirements.txt

# 3. API key
cp .env.example .env
#    then edit .env and paste your key:
#    ANTHROPIC_API_KEY=sk-ant-...

# 4. Build the RAG vector index from data/docs/
python -m rag.embed

# 5. Run the server
python -m uvicorn main:app --port 8000
```

Verify:

```bash
curl http://localhost:8000/healthz
# → {"ok": true, "parts_loaded": 40}
```

> Skip `--reload` during normal use. Its file watcher sees Python's
> `__pycache__` writes and the embedder's cache files and can enter a
> restart loop mid-request. If you want reload while editing, scope it:
> `--reload --reload-dir agent --reload-dir rag --reload-include "*.py"`.

---

## Data pipeline

The app reads only from the local store — it never hits PartSelect at request
time. Two one-off scripts produce that store:

```bash
python scrape.py      # → data/parts.json + data/docs/*.txt   (≈40 parts)
python -m rag.embed   # → data/vectors.npy + data/vectors_meta.json
```

**`scrape.py`** pulls ~40 parts: the named example `PS11752778`, every part
listed under dishwasher model `WDT780SAEM1`, plus three more model pages to
balance fridge/dishwasher coverage. PartSelect sits behind Akamai, which rejects
plain `requests`/`curl` on a TLS fingerprint; `curl_cffi` impersonating Chrome
gets through. The pages are server-rendered HTML, so no headless browser is
needed. Per part it captures: PS#, manufacturer part #, name, price, in-stock,
brand, fits-brands, description, symptoms fixed, replaced part numbers, ~10
sample compatible models, install difficulty, install video ID, image URL, and
a few customer repair stories. The committed `data/` means you don't need to
re-scrape to run the project.

**`rag/embed.py`** embeds each `data/docs/*.txt` with `all-MiniLM-L6-v2`
(local, 384-dim) and writes an L2-normalized matrix plus a parallel metadata
list. Re-run it whenever the data changes.

---

## Tests

```bash
# Guardrail unit tests — fast, no API calls
python -m pytest tests/ -v

# Golden eval harness — hits the live Anthropic API (~$0.10, ~2 min)
python -m pytest evals/golden.py -v
```

The 12 golden cases cover: direct lookup, compatibility (clear mismatch,
likely-compatible, unknown), symptom search, repair guide, entity memory across
turns ("this part"), MPN lookup, appliance-type filtering, a multi-tool flow,
out-of-scope refusal, and honest handling of nonexistent parts. Each case
asserts the **right tool was called**, not just that some text came back —
mocking the LLM would only test the dispatcher, so the harness calls the real
model on purpose.

---

## Key design decisions & trade-offs

- **JSON store, not a database.** The catalog is ~40 read-only records. SQLite
  or Postgres would be ceremony with no benefit at this scale, and a JSON file
  is reviewable and makes the eval harness deterministic. The store is wrapped
  behind `PartStore`, so swapping in a DB is a one-file change — every tool
  callsite is unaffected.

- **NumPy cosine similarity, not FAISS.** At 40 documents a single matrix
  multiply is faster than building and loading a FAISS index, with one fewer
  binary dependency. FAISS pays off around 100k+ vectors; below that it's
  premature. The metadata filter (appliance type) runs *before* ranking so an
  appliance filter that excludes everything returns empty rather than a
  best-of-irrelevant match.

- **Local embeddings, not a hosted API.** `all-MiniLM-L6-v2` needs no second
  API key and costs nothing per query — right-sized for this corpus.

- **Validation after the draft, not constrained generation.** The agent writes a
  full answer; only the final text is gated. This is cheaper, simpler to test,
  and the guardrail rules are plain functions with their own unit tests.

- **Entity memory via regex in the system prompt, not a tool.** Resolving "this
  part" with a regex over the conversation costs microseconds; a dedicated
  resolution tool would burn an LLM round on every pronoun. The trade-off is
  that genuinely ambiguous references ("the second one you mentioned") aren't
  handled — acceptable at this scope.

---

## Known limitations / what I'd do next

- **Sessions are in-memory, single-process.** Restarting the server clears
  history; running multiple uvicorn workers would fragment it. Production swap:
  move `SessionStore` to Redis so workers share state and sessions survive
  restarts. The agent loop and FastAPI layer don't change.

- **History is bounded by turn count, not tokens.** A token-budget trim
  (`tiktoken` / count-tokens endpoint) would be more precise.

- **Data freshness requires a re-scrape + restart.** Prices and stock can drift.
  Options, in order of effort: a scheduled re-scrape (cron/Action), a per-record
  TTL with on-demand refresh, or — in a real deployment — replacing the scraper
  with PartSelect's own price/inventory API behind the same tool interface.

- **Guardrail is pattern-based.** It catches the high-value failure (ungrounded
  compatibility claims) with tight regexes and a proximity check. A model-graded
  check would catch more paraphrases at the cost of latency and an extra call.

---

## Project layout

```
main.py                FastAPI app: POST /chat, GET /healthz, CORS, card payload
scrape.py              One-off scraper (curl_cffi + BeautifulSoup)
agent/
  store.py             parts.json reader, O(1) lookup by PS# or MPN
  context.py           session history + entity memory
  tools.py             tool schemas + implementations + dispatcher
  prompts.py           system prompt (scope, tool discipline, pricing honesty)
  loop.py              Claude tool-use loop + guardrail integration
  guardrails.py        post-draft validation (the two invariants)
rag/
  embed.py             one-off: docs/ → vectors.npy
  retrieve.py          cosine search with metadata filter + threshold
evals/
  golden.py            12 golden cases
  runner.py            harness that drives the real agent loop
tests/
  test_guardrails.py   guardrail unit tests
data/
  parts.json           structured source of truth (~40 parts)
  docs/                one prose blob per part, embedded for RAG
  vectors.npy          embedding matrix
  vectors_meta.json    per-vector metadata
```
