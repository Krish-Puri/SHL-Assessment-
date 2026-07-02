# SHL Assessment Recommender — Approach Document

## 1. Design Choices

### Why a hybrid retrieval + LLM agent?
The evaluation is Recall@10 — the agent must surface the right assessments from 377 options.
A pure LLM (even with high reasoning quality) cannot reliably recall from a catalog it wasn't
trained on. Combining FAISS semantic search (high recall on relevance) with keyword attribute
filters (precision on level/type/language) gives the best trade-off.

### Why Groq over other providers?
Groq's free tier offers `llama-3.3-70b` with sub-second latency — fast enough for a 30s
API timeout. The conversational agent's job is primarily routing and short-list construction,
not complex reasoning, so this model is well-sized.

### Architecture: stateless API
Every `/chat` request carries the full conversation history. No per-conversation state on the
server. This matches the eval harness design (replay with personas) and makes horizontal
scaling trivial.

### Python 3.12 + FastAPI
Standard scientific-Python stack. FastAPI gives Pydantic schema enforcement and async HTTP
out-of-the-box. Uvicorn runs the server.

---

## 2. Retrieval Setup

### Catalog preprocessing (`catalog_indexer.py`)
- Raw JSON scraped from SHL (with malformed raw newlines in some `name` fields) is loaded
  with `json.loads(strict=False)`.
- Each of 377 items is normalized: test-type codes (K/A/P/S/B/C/D), job-level booleans
  (`is_senior`, `is_graduate`, etc.), and tech-keyword extraction from name+description.
- Two indexes are built at startup (FAISS index is always rebuilt fresh to keep memory footprint low):
  - FAISS `IndexFlatIP` (128-dim TF-IDF+SVD embeddings, cosine similarity) — built from
    `TfidfVectorizer` (5000 features, 1-2 ngrams, sublinear TF) + `TruncatedSVD` (128 dims)
  - BM25 (`rank_bm25.BM25Plus`) over name + description

### Three-stage hybrid retrieval (BM25 + FAISS + RRF)
The retrieval pipeline has three stages:
1. **BM25 search** — exact-match friendly for entities like "Java 8", "OPQ32r", ".NET"
2. **FAISS semantic search** — TF-IDF+SVD vector similarity (e.g. "communicative leader" → personality)
3. **Reciprocal Rank Fusion (RRF)** — merges both rankings with `k=60` smoothing
4. **Attribute post-filter** — filters by level/type/language, preserving fused rank order

This replaces pure-FAISS retrieval: BM25 catches entity abbreviations that embeddings miss.

Level matching maps conversational signals to catalog semantics:
- `"senior"` → `is_senior=True` OR "Mid-Professional" / "Professional Individual Contributor"
- `"graduate"` → `is_graduate=True` OR "Graduate" in job_levels
- `"entry"` → `is_entry=True` OR "Entry-Level" in job_levels

### No LLM mode (demo / cold-start)
When `GROQ_API_KEY` is absent, a rule-based fallback uses keyword-only retrieval directly,
without calling the LLM. This lets the service function for local testing and cold-start
hosting evaluation.

---

## 3. Prompt Design

### LLM Intent Classification (with regex fallbacks)
A single cheap LLM call classifies the user's intent into one of: `CLARIFY`, `RECOMMEND`,
`REFINE_ADD`, `REFINE_DROP`, `REFINE_REPLACE`, `COMPARE`, `REFUSE`, `CONFIRM`. This is robust
to phrasal variants like "contrast X with Y" (which would miss a regex for "difference between").
Regex fallbacks are layered on top for high-confidence patterns the LLM sometimes misclassifies:
DROP ("drop the X test"), ADD ("also include X"), and REPLACE ("make that X instead"). One
~50-token Groq call per turn plus O(1) regex checks.

### Confidence-based recommendation
After intent routing, `ConversationContext.confidence_score()` counts distinct high-information
signals (level, domain, purpose, 2+ test types, 2+ tech keywords) to decide whether to
recommend or clarify. No more turn-count forcing — if confidence is high enough, we recommend.

**System prompt** establishes behavioral rules:
1. Never invent — only recommend from catalog
2. Every URL must be an exact catalog link
3. Ask one clarifying question if context is insufficient
4. When the user refines, update the list — don't restart
5. When comparing, retrieve both from catalog and compare
6. Refuse off-topic / non-SHL requests
7. Always include the full current shortlist in recommendations
8. Respond with valid JSON only
9. When replacing ("make that X instead"), discard the old technology/level and re-retrieve with the new one

**User turn prompt** passes:
- Last 8 messages of conversation history
- Extracted context flags (level, domain, test types, tech keywords, languages, current shortlist)
- Confidence-aware reply variation (high/medium/low confidence each get different phrasing)

The LLM is instructed to output a single JSON block with `reply`, `recommendations[]`,
and `end_of_conversation`. JSON is parsed and validated; URLs are re-checked against the
catalog before being returned.

---

## 4. Evaluation Approach

### Unit tests (44 tests, all passing)
- Catalog loading: 377 items, all URLs valid
- Key normalization: K/A/P/S/B/C/D codes
- Level flags: senior/graduate/entry/mid booleans
- Attribute filters: level, test type, language, tech keyword
- BM25 + FAISS + RRF hybrid retrieval
- LLM intent classifier: CLARIFY/RECOMMEND/REFINE_ADD/REFINE_DROP/COMPARE/REFUSE/CONFIRM
- Confidence scoring: ctx.confidence_score() returns correct values
- Off-topic rejection via LLM intent classification
- Mid-conversation refinement: add/drop detection, shortlist preservation
- Schema compliance: empty/10-item lists, end_of_conversation flag, auto-truncation

### Replay harness (`eval/harness.py`)
20 synthetic test cases covering:
- Vague initial query → CLARIFY
- Specific requests → RECOMMEND (with ground-truth Recall@10)
- "contrast X with Y" / "difference between" → COMPARE (LLM-classified)
- Prompt injection → REFUSE
- Shortlist refinement → REFINE_ADD / REFINE_DROP
- Confirmation → CONFIRM

Metrics measured: Recall@10, hallucination rate, schema compliance, latency.

### Behavioral probes
- **Off-topic rejection**: LLM classifier (not regex) catches "contrast OPQ with GSA",
  prompt injection, legal questions, salary questions.
- **Hallucination prevention**: All returned URLs are re-validated against `find_by_name()`.
  Items not found in catalog are dropped.
- **Confidence-based recommendation**: No turn-count forcing — recommends when
  `confidence_score() >= 1` and retrieves results, otherwise asks a clarifying question.

---

## 5. What Didn't Work (and How I Fixed It)

### Bug: FAISS pre-filter → zero recall for Java+senior
The catalog's "senior" items use "Mid-Professional" in job_levels, not the string "senior".
Pre-filtering the catalog before semantic search returned 0 candidates.
**Fix**: Post-filter semantic results instead — semantic search runs over all 377 items,
attribute filters apply to the ranked candidate list.

### Bug: Dual module-level EMBEDDER globals
Both `agent.py` and `catalog_indexer.py` defined `EMBEDDER = None` at module level.
`_load_catalog()` in agent.py set `agent.EMBEDDER`, but `retrieve_assessments()` read
from `agent.EMBEDDER` (still `None`). Semantic search silently returned 0.
**Fix**: `_load_catalog()` now sets `catalog_indexer.EMBEDDER` and `catalog_indexer.FAISS_INDEX`
so all consumers read from the same globals. Used `import catalog_indexer` (not import-from)
to avoid local shadowing.

### Bug: Catalog JSON had raw unescaped newlines
The scraped JSON had literal newlines inside string values (e.g., `"name": "Microsoft \n 365"`).
Standard `json.loads` raised `JSONDecodeError`.
**Fix**: `json.loads(text, strict=False)` — Python's strict=False accepts control characters
inside strings, which is the right behavior for this broken-scrape input.

### Bug: BM25 ZeroDivisionError when CATALOG not loaded
After adding BM25 startup to `load_index()`, tests that called `load_index()` directly
(before running `main()`) found `CATALOG = []`, causing `rank_bm25` to divide by zero.
**Fix**: `load_index()` now loads the catalog lazily if `CATALOG` is empty before building
the BM25 index.

### Bug: Off-topic detection missed weather / sports / generic questions
`is_off_topic()` relied only on explicit prompt-injection patterns. Questions like "What's the
weather in Delhi?" had no injection signal, passed through to the LLM, and were misclassified
as CLARIFY/RECOMMEND.
**Fix**: Three-layer protection — (1) explicit off-topic patterns (weather, sports, news, etc.),
(2) `has_shl_signal()` — if any SHL-related keyword is found (tech, domain, assessment type,
seniority), the message is on-topic regardless of phrasing, (3) generic-question guard —
questions matching `^\s*(what|how|when|where|who|why|can you|tell me).*\?\s*$` without an SHL
signal are refused.

### Bug: REFINE_ADD bypassed by LLM misclassification
The LLM classified "Also include personality assessments." as RECOMMEND instead of REFINE_ADD,
bypassing `_handle_refinement` entirely. The refinement handler was correct — it just never
ran because routing never reached it.
**Fix**: Added a `REFINE_ADD` regex fallback at the routing layer (same pattern used for DROP
and CONFIRM), so high-confidence ADD phrases are caught before LLM classification.

### Bug: REFINE_ADD returned zero results even when handler ran
When adding a test type (e.g. "personality"), the code appended "personality" to
`ctx.tech_keywords` on top of the existing `["java", "spring", "backend"]`. The BM25/FAISS
query required all four terms simultaneously — no catalog item has "java AND spring AND
personality" — so retrieval returned 0 results.
**Fix**: When `any_test_type_added` is True, clear `ctx.tech_keywords` for the ADD retrieval
only (level + test_type drive the query). Applied in both the populated-shortlist path
(and the empty-shortlist/reconstruction path).

### Bug: sentence-transformers OOM on Render free tier
Render's free tier has 512MB RAM. `sentence-transformers` + `torch` + the model files
(~400MB) exhausts memory on the first cold start, before the app even begins serving requests.
**Fix**: Replaced with `scikit-learn` TF-IDF (`TfidfVectorizer`) + `TruncatedSVD` (128 dims).
Memory footprint dropped from ~450MB to ~60MB. TF-IDF also has better exact-term matching
for entities like "Java 8" or "OPQ32r", though conceptually it is less rich than BERT embeddings.
BM25 already covers exact-match well; TF-IDF+SVD supplements it with latent semantic themes.
"Actually make that Python developers instead." had no matching ADD or DROP pattern, so it
fell through to the LLM and was misclassified. Java results were still returned.
**Fix**: Added `REFINE_REPLACE` intent with its own regex pattern and `_handle_replacement()`
handler that extracts the new tech keyword, updates `ctx.tech_keywords` and `ctx.level`,
resets domain, and re-retrieves.

### AI tools used
Claude was used as an AI-assisted development tool for scaffolding, debugging, and generating initial test templates. All architectural decisions, retrieval design, hybrid search implementation, evaluation methodology, and final code modifications were reviewed, implemented, and validated manually.
