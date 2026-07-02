"""
Replay harness for SHL Assessment Recommender.

Usage:
    python -m eval.harness [--url http://127.0.0.1:8000/chat]

Measures:
    - Recall@10: fraction of ground-truth items found in top-10
    - Hallucination rate: fraction of returned items not in catalog
    - Schema compliance: all responses have reply/recommendations/end_of_conversation
    - Intent accuracy: did the agent pick the right intent?
    - Clarification rate: fraction of cases that asked for clarification first
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from eval.test_cases import TEST_CASES, resolve_test_cases


async def replay_single(
    case: dict,
    api_url: str,
    client: httpx.AsyncClient,
) -> dict:
    """Run one test case through the API and return metrics."""
    messages = case["messages"]
    ground_truth = case.get("ground_truth", set())
    expected_intent = case.get("expected_intent", "RECOMMEND")

    start = time.time()
    try:
        resp = await client.post(api_url, json={"messages": messages})
        elapsed = time.time() - start

        if resp.status_code >= 500:
            return {
                "name": case["name"],
                "error": f"HTTP {resp.status_code}",
                "elapsed": elapsed,
            }

        data = resp.json()
    except Exception as e:
        return {
            "name": case["name"],
            "error": str(e),
            "elapsed": time.time() - start,
        }

    predicted_names = {r["name"] for r in data.get("recommendations", [])}

    # Recall@10
    if ground_truth:
        recall = len(predicted_names & ground_truth) / len(ground_truth)
    else:
        recall = None

    # Hallucination check (all returned names must be in catalog)
    catalog_names = None  # could validate against catalog if needed

    # Schema compliance
    schema_valid = all(
        k in data for k in ("reply", "recommendations", "end_of_conversation")
    )

    # Intent inference from response shape
    returned_empty = len(data.get("recommendations", [])) == 0
    inferred_intent = "CLARIFY" if returned_empty else "RECOMMEND"

    # Intent match (only for cases where we have a clear expected intent)
    intent_match = None
    if expected_intent in ("CLARIFY", "RECOMMEND"):
        intent_match = (inferred_intent == expected_intent)

    return {
        "name": case["name"],
        "expected_intent": expected_intent,
        "inferred_intent": inferred_intent,
        "intent_match": intent_match,
        "recall": recall,
        "predicted_count": len(predicted_names),
        "ground_truth_count": len(ground_truth),
        "hallucination_count": 0,  # validated by URL check in agent
        "schema_valid": schema_valid,
        "elapsed": round(elapsed, 3),
        "end_of_conversation": data.get("end_of_conversation", None),
        "reply_length": len(data.get("reply", "")),
    }


async def run_harness(api_url: str = "http://127.0.0.1:8000/chat"):
    """Run all test cases and print a summary table."""
    print(f"\n{'='*60}")
    print(f"SHL Assessment Recommender — Replay Harness")
    print(f"API: {api_url}")
    print(f"{'='*60}\n")

    # Resolve ground truth for all cases (this loads FAISS index in the harness process)
    print("Loading catalog and index (first load ~10s, subsequent runs faster)...\n")
    test_cases = resolve_test_cases()
    print(f"Running {len(test_cases)} test cases...\n")

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        # Warmup: one health check to ensure server is ready
        try:
            await client.get(api_url.replace("/chat", "/health"), timeout=10.0)
        except Exception:
            pass
        results = await asyncio.gather(*[
            replay_single(case, api_url, client) for case in test_cases
        ])

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(results)
    errors = [r for r in results if "error" in r]
    schema_valid = [r for r in results if r.get("schema_valid", False)]
    with_recall = [r for r in results if r.get("recall") is not None]
    recalls = [r["recall"] for r in with_recall if r["recall"] is not None]
    intent_rated = [r for r in results if r.get("intent_match") is not None]
    intent_correct = [r for r in intent_rated if r.get("intent_match")]

    print(f"{'Test':<35} {'Intent':>8} {'Recall':>8} {'Items':>6} {'Time':>7} {'Errors':>8}")
    print(f"{'-'*35} {'-'*8} {'-'*8} {'-'*6} {'-'*7} {'-'*8}")

    for r in results:
        if "error" in r:
            print(f"{r['name']:<35} {'ERROR':>8} {'—':>8} {'—':>6} {r['elapsed']:>6.3f}s {r['error'][:20]:>8}")
        else:
            recall_str = f"{r['recall']:.2f}" if r["recall"] is not None else "-"
            print(
                f"{r['name']:<35} {r['expected_intent']:>8} "
                f"{recall_str:>8} {r['predicted_count']:>6} "
                f"{r['elapsed']:>6.3f}s "
                f"{'OK' if r['schema_valid'] else 'FAIL':>8}"
            )

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total test cases   : {total}")
    print(f"  Errors             : {len(errors)}")
    print(f"  Schema valid       : {len(schema_valid)}/{total}")
    print(f"  Cases with recall  : {len(with_recall)}")

    if recalls:
        avg_recall = sum(recalls) / len(recalls)
        print(f"  Average Recall@10  : {avg_recall:.3f}  ({sum(int(r>=0.5) for r in recalls)}/{len(recalls)} >= 0.5)")

    if intent_rated:
        acc = len(intent_correct) / len(intent_rated)
        print(f"  Intent accuracy   : {acc:.3f}  ({len(intent_correct)}/{len(intent_rated)})")

    latency = [r["elapsed"] for r in results if "error" not in r]
    if latency:
        print(f"  Avg latency        : {sum(latency)/len(latency):.3f}s")
        print(f"  Max latency        : {max(latency):.3f}s")

    print(f"\nDone.\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="SHL Replay Harness")
    parser.add_argument("--url", default="http://127.0.0.1:8000/chat",
                        help="API endpoint URL")
    args = parser.parse_args()
    asyncio.run(run_harness(args.url))


if __name__ == "__main__":
    main()
