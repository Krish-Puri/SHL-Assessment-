# SHL Assessment Recommender

**Live API:** https://shl-assessment-1-va6q.onrender.com

A conversational agent that recommends SHL assessments from a catalog of 377 products. It uses hybrid retrieval (BM25 + TF-IDF/SVD + Reciprocal Rank Fusion) and Groq's `llama-3.3-70b-versatile` model to route user intents and construct a shortlist of relevant assessments.

## Intent Types

| Intent | Description |
|--------|-------------|
| `RECOMMEND` | Recommend assessments matching the user's hiring scenario |
| `CLARIFY` | Ask a clarifying question when context is insufficient |
| `REFINE_ADD` | Add a test type or category to the current shortlist |
| `REFINE_DROP` | Remove a test type from the current shortlist |
| `REFINE_REPLACE` | Replace a technology or level (e.g. "make that Python instead") |
| `COMPARE` | Compare two specific assessments |
| `CONFIRM` | Confirm the current shortlist |
| `REFUSE` | Refuse off-topic or out-of-scope requests |

## API

**POST /chat**

```json
{
  "messages": [
    { "role": "user", "content": "I need assessments for senior Java developers" }
  ]
}
```

**Response:**

```json
{
  "reply": "Here are three recommendations for senior Java developers...",
  "recommendations": [
    {
      "name": "Java 8 Programming Assessment",
      "link": "https://www.shl.com/shl-...",
      "type": "K",
      "duration": "40 mins"
    }
  ],
  "end_of_conversation": false
}
```

**GET /health** — health check endpoint.

## Architecture

- **FastAPI** + Uvicorn, Python 3.12
- **Hybrid retrieval:** BM25 (exact-match) + TF-IDF/SVD 128-dim (semantic) + Reciprocal Rank Fusion
- **Stateless API:** every request carries full conversation history
- **Groq LLM** (`llama-3.3-70b-versatile`) for intent routing and response generation
- **Render free tier** deployment (cold-start ~30-60s; always-on after first request)

## Setup

```bash
pip install -r requirements.txt
python -m catalog_indexer          # build indexes (one-time)
python api.py                       # start server at http://127.0.0.1:8000
```

## Testing

```bash
python -m pytest tests/ -v          # unit tests
python -m eval.harness              # behavioral replay harness (requires live server)
```

## Files

| File | Description |
|------|-------------|
| `api.py` | FastAPI app, `/chat` and `/health` endpoints |
| `agent.py` | Intent routing, context extraction, refinement handlers |
| `catalog_indexer.py` | Catalog normalization, BM25 + FAISS index building |
| `retrieval.py` | Hybrid retrieval functions |
| `eval/harness.py` | Behavioral replay harness |
| `tests/` | Unit and integration tests |
| `approach.pdf` | 2-page approach summary |
