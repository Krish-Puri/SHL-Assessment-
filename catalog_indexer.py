"""
Catalog indexer: loads raw SHL catalog, normalizes it,
and builds a FAISS index for semantic search + keyword filtering.
"""
import json
import re
import os
from pathlib import Path
from typing import Any, Union, Optional, List

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    HAS_LANG = True
except ImportError as e:
    HAS_LANG = False
    print(f"[WARN] faiss/sentence-transformers not available: {e}. Using keyword-only retrieval.")


CATALOG_RAW = Path(__file__).parent / "shl_product_catalog.json"
CATALOG_OUT = Path(__file__).parent / "catalog_processed.json"
INDEX_FILE = Path(__file__).parent / "catalog_index.faiss"
EMBEDDER_NAME = "all-MiniLM-L6-v2"

# Module-level globals (set by load_index / build_faiss_index)
EMBEDDER = None
FAISS_INDEX = None
BM25_INDEX = None
CATALOG: list = []

# ── Raw catalog loading ───────────────────────────────────────────────────────

def load_raw_catalog(path: Union[str, Path] = CATALOG_RAW) -> list:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f, strict=False)


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize_key(key: str) -> str:
    """Map a 'keys' value to its single-letter code."""
    mapping = {
        "Knowledge & Skills": "K",
        "Ability & Aptitude": "A",
        "Personality & Behavior": "P",
        "Simulations": "S",
        "Biodata & Situational Judgment": "B",
        "Competencies": "C",
        "Assessment Exercises": "E",
        "Development & 360": "D",
    }
    return mapping.get(key, key)


def compute_level_flags(levels: list[str]) -> dict[str, bool]:
    """Derive boolean flags from job_levels list."""
    all_levels = " ".join(levels).lower()
    return {
        "is_entry": bool(re.search(r"entry.level|front.line.manager", all_levels)),
        "is_graduate": "graduate" in all_levels,
        "is_mid": bool(re.search(r"mid.professional|professional.individual.contributor", all_levels)),
        "is_senior": bool(re.search(r"manager|director|executive|supervisor", all_levels)),
        "is_general": "general population" in all_levels,
    }


def extract_tech_keywords(name: str, description: str) -> list[str]:
    """Extract technology/skill keywords from name + description for keyword matching."""
    tech_patterns = [
        "java", "python", "sql", "mysql", "postgresql", "mongodb",
        "aws", "azure", "docker", "kubernetes", "k8s",
        "spring", "django", "flask", "react", "angular", "vue",
        "c++", "c#", ".net", "rust", "golang", "go",
        "excel", "word", "powerpoint", "outlook",
        "linux", "unix", "windows server",
        "sap", "salesforce", "tableau", "power bi",
        "agile", "scrum", "jira",
        "html", "css", "javascript", "typescript",
        "rest", "api", "microservice", "microservices",
        "cloud", "devops", "ci/cd", "jenkins",
        "machine learning", "ml", "ai", "data science",
        "networking", "security", "hipaa", "finance",
        "accounting", "statistics", "financial analysis",
        "hr", "human resources", "recruitment",
        "sales", "customer service", "contact center",
        "healthcare", "nursing", "medical",
        "safety", "manufacturing", "industrial",
        "project management", "leadership",
        "communication", "business writing",
    ]
    text = (name + " " + description).lower()
    found = []
    for t in tech_patterns:
        if t in text:
            found.append(t)
    return found


def normalize_item(raw: dict) -> dict:
    """Transform a raw catalog entry into a normalized form."""
    keys_list = raw.get("keys", [])
    test_type_codes = [normalize_key(k) for k in keys_list]

    level_flags = compute_level_flags(raw.get("job_levels", []))
    tech_kw = extract_tech_keywords(
        raw.get("name", ""), raw.get("description", "")
    )

    return {
        "entity_id": raw.get("entity_id", ""),
        "name": raw.get("name", ""),
        "link": raw.get("link", ""),
        "keys": keys_list,
        "test_type_codes": test_type_codes,
        "job_levels": raw.get("job_levels", []),
        "languages": raw.get("languages", []),
        "duration": raw.get("duration", ""),
        "description": raw.get("description", ""),
        "remote": raw.get("remote", ""),
        "adaptive": raw.get("adaptive", ""),
        **level_flags,
        "tech_keywords": tech_kw,
    }


def preprocess_catalog(raw_catalog: list[dict]) -> list[dict]:
    """Normalize all catalog entries."""
    return [normalize_item(item) for item in raw_catalog]


# ── FAISS index builder ────────────────────────────────────────────────────────

def build_faiss_index(catalog: list, index_path: Union[str, Path] = INDEX_FILE):
    """Build and save FAISS index from normalized catalog embeddings."""
    global EMBEDDER, FAISS_INDEX
    if not HAS_LANG:
        print("[SKIP] FAISS indexing skipped (deps not installed).")
        return

    embedder = SentenceTransformer(EMBEDDER_NAME)

    texts = [
        f"{item['name']} | {item['description']} | types: {','.join(item['keys'])}"
        for item in catalog
    ]
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=64)

    # Normalize to unit length so IndexFlatIP == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine similarity for normalized vectors
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, str(index_path))
    EMBEDDER = embedder
    FAISS_INDEX = index
    print(f"[OK] FAISS index built: {len(catalog)} items, dim={dim}, IndexFlatIP (cosine)")


def load_index(index_path: Union[str, Path] = INDEX_FILE):
    """Load existing FAISS index and initialise the embedder. Sets module globals."""
    global EMBEDDER, FAISS_INDEX, BM25_INDEX
    if not HAS_LANG or not os.path.exists(index_path):
        return None
    idx = faiss.read_index(str(index_path))
    EMBEDDER = SentenceTransformer(EMBEDDER_NAME)
    FAISS_INDEX = idx
    # Ensure CATALOG is populated before building BM25 (may be called before main())
    global CATALOG
    if not CATALOG:
        raw = load_raw_catalog()
        CATALOG = preprocess_catalog(raw)
    build_bm25_index(CATALOG)
    return idx


# ── BM25 index builder ─────────────────────────────────────────────────────────

def build_bm25_index(catalog: list[dict], k1: float = 1.5, b: float = 0.75):
    """Build and store a BM25 index over name + description for each catalog item."""
    global BM25_INDEX
    try:
        from rank_bm25 import BM25Plus
    except ImportError:
        print("[WARN] rank-bm25 not installed, BM25 search unavailable.")
        return None

    tokenized_corpus = [
        (item["name"] + " " + item["description"]).lower().split()
        for item in catalog
    ]
    BM25_INDEX = BM25Plus(tokenized_corpus, k1=k1, b=b)
    print(f"[OK] BM25 index built for {len(catalog)} items")
    return BM25_INDEX


def bm25_search(query: str, catalog: list[dict], top_k: int = 40) -> list[dict]:
    """BM25 search returning top-k catalog items sorted by relevance score."""
    if BM25_INDEX is None:
        return []
    tokens = query.lower().split()
    scores = BM25_INDEX.get_scores(tokens)
    # Sort indices by score descending, take top-k with positive scores
    indexed_scores = sorted(
        [(i, scores[i]) for i in range(len(scores)) if scores[i] > 0],
        key=lambda x: x[1],
        reverse=True
    )
    return [catalog[i] for i, _ in indexed_scores[:top_k]]


# ── Retrieval functions ────────────────────────────────────────────────────────

def semantic_search(query: str, catalog: list[dict], embedder, index, top_k: int = 15) -> list[dict]:
    """ANN search in FAISS index."""
    if index is None or embedder is None:
        return []
    q_emb = embedder.encode([query])
    _, indices = index.search(q_emb.astype(np.float32), top_k)
    return [catalog[i] for i in indices[0] if i < len(catalog)]


def keyword_filter(
    catalog: list[dict],
    test_types: list[str] | None = None,
    levels: list[str] | None = None,
    languages: list[str] | None = None,
    tech_keywords: list[str] | None = None,
    exclude_names: list[str] | None = None,
) -> list[dict]:
    """Simple keyword / attribute pre-filter over the full catalog.

    levels: list of level names (e.g. ["senior", "graduate"]).
    Special handling:
      - "senior" maps to is_senior=True OR Mid-Professional OR Professional Individual Contributor
      - "graduate" maps to is_graduate=True OR Entry-Level
      - "mid" maps to is_mid=True OR Mid-Professional
    """
    results = []
    for item in catalog:
        if test_types:
            if not any(tc in item.get("test_type_codes", []) for tc in test_types):
                continue
        if levels:
            level_match = False
            for lvl in levels:
                lvl_lower = lvl.lower()
                if lvl_lower == "senior":
                    # Senior: is_senior flag (Manager/Director/Executive) OR Mid-Professional / PIC
                    # BUT exclude items that have Entry-Level in job_levels (catalog inconsistency fix)
                    job_levels = item.get("job_levels", [])
                    if "Entry-Level" in job_levels:
                        continue  # entry-level items should not appear in senior results
                    if item.get("is_senior") or "Mid-Professional" in job_levels or "Professional Individual Contributor" in job_levels:
                        level_match = True
                        break
                elif lvl_lower == "graduate":
                    if item.get("is_graduate") or "Graduate" in item.get("job_levels", []):
                        level_match = True
                        break
                elif lvl_lower == "mid":
                    if item.get("is_mid") or "Mid-Professional" in item.get("job_levels", []) or "Professional Individual Contributor" in item.get("job_levels", []):
                        level_match = True
                        break
                elif lvl_lower == "entry" or lvl_lower == "entry-level":
                    if item.get("is_entry") or "Entry-Level" in item.get("job_levels", []):
                        level_match = True
                        break
                else:
                    # Exact match on job_levels string
                    if any(lvl_lower in l.lower() for l in item.get("job_levels", [])):
                        level_match = True
                        break
            if not level_match:
                continue
        if languages:
            item_langs = [l.lower() for l in item.get("languages", [])]
            if not any(lang.lower() in item_langs for lang in languages):
                continue
        if tech_keywords:
            # Match if keyword appears in EITHER the tech_keywords field OR in name/description
            # (treating tech_keywords as broad content search, not just job-specific skills)
            item_kw = [k.lower() for k in item.get("tech_keywords", [])]
            item_text = (item["name"] + " " + item.get("description", "")).lower()
            if not any(kw.lower() in item_kw or kw.lower() in item_text for kw in tech_keywords):
                continue
        if exclude_names:
            if any(ex.lower() in item["name"].lower() for ex in exclude_names):
                continue
        results.append(item)
    return results


def hybrid_retrieve(
    query: str,
    catalog: list[dict],
    embedder,
    index,
    test_types: list[str] | None = None,
    levels: list[str] | None = None,
    languages: list[str] | None = None,
    tech_keywords: list[str] | None = None,
    exclude_names: list[str] | None = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Three-stage hybrid retrieval:
      1. BM25 exact-match search (good for entities like "Java 8", "OPQ32r")
      2. FAISS semantic search (conceptual similarity)
      3. Reciprocal Rank Fusion to merge both rankings
      4. Attribute post-filter (preserving fused rank order)
      5. Return top-k results
    """
    # Retrieve more candidates when test_types filter is active — post-filter will remove
    # many candidates, so start with a larger pool to avoid empty results
    k_initial = top_k * 20 if test_types else top_k * 4  # 200 or 40 candidates

    # Stage 1: BM25 candidates
    bm25_candidates = bm25_search(query, catalog, top_k=k_initial)
    bm25_scores = {}
    for rank, item in enumerate(bm25_candidates):
        bm25_scores[item["name"]] = len(bm25_candidates) - rank  # higher = better

    # Stage 2: FAISS semantic candidates
    if index is not None and embedder is not None and query:
        sem_candidates = semantic_search(query, catalog, embedder, index, top_k=k_initial)
        sem_scores = {}
        for rank, item in enumerate(sem_candidates):
            sem_scores[item["name"]] = len(sem_candidates) - rank  # higher = better
    else:
        sem_candidates = []
        sem_scores = {}

    # Stage 3: Reciprocal Rank Fusion
    all_names = list({item["name"] for item in bm25_candidates + sem_candidates})
    rrf_scores = {}
    k_rrf = 60  # standard RRF smoothing parameter
    for name in all_names:
        r = 0.0
        if name in bm25_scores:
            r += 1.0 / (k_rrf + bm25_scores[name])
        if name in sem_scores:
            r += 1.0 / (k_rrf + sem_scores[name])
        rrf_scores[name] = r

    fused_rank = sorted(all_names, key=lambda n: rrf_scores[n], reverse=True)
    name_to_item = {item["name"]: item for item in catalog}
    fused_candidates = [name_to_item[n] for n in fused_rank if n in name_to_item]

    # Stage 4: Attribute post-filter (preserving RRF order)
    filtered = keyword_filter(
        fused_candidates,
        test_types=test_types,
        levels=levels,
        languages=languages,
        tech_keywords=tech_keywords,
        exclude_names=exclude_names,
    )
    return filtered[:top_k]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading raw catalog...")
    raw = load_raw_catalog()
    print(f"  Loaded {len(raw)} items")

    print("Normalizing...")
    catalog = preprocess_catalog(raw)

    out = CATALOG_OUT
    with open(out, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    print(f"  Normalized catalog -> {out}")

    if HAS_LANG:
        print("Building FAISS index (this may take ~30s)...")
        build_faiss_index(catalog)
        print("Building BM25 index...")
        build_bm25_index(catalog)
    else:
        print("FAISS index not built (dependencies missing).")

    print("Done.")


if __name__ == "__main__":
    main()
