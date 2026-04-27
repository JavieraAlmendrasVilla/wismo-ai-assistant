"""
run_evals.py
------------
Runs all 50 eval cases through the WISMO chain and computes:
  1. Answer Relevance        — % containing expected_answer_contains strings
  2. Not-Found Handling      — % of NOT_FOUND cases gracefully handled
  3. Guardrail Trigger Rate  — % of responses that triggered output_validator
  4. Avg Latency (ms)
  5. Answer Faithfulness     — % of answers not flagged by the output validator

Results saved to evaluation/results.json and printed as a summary table.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"))
logger = logging.getLogger(__name__)

EVAL_DATASET_PATH = ROOT / "evaluation" / "eval_dataset.json"
RESULTS_PATH = ROOT / "evaluation" / "results.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_eval_dataset() -> list[dict]:
    """Load the 50-case eval dataset."""
    with open(EVAL_DATASET_PATH) as f:
        return json.load(f)


def relevance_check(answer: str, expected_contains: list[str]) -> bool:
    """Return True if the answer contains ALL expected substrings (case-insensitive)."""
    answer_lower = answer.lower()
    return all(phrase.lower() in answer_lower for phrase in expected_contains)


def not_found_check(answer: str) -> bool:
    """Return True if the answer gracefully handles a not-found case."""
    not_found_phrases = [
        "not found", "couldn't find", "could not find", "no tracking",
        "unable to locate", "verify", "no record", "no shipment",
        "don't have", "do not have", "no information",
    ]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in not_found_phrases)


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_evals() -> dict[str, Any]:
    """
    Execute all eval cases and return a results dict.

    Returns:
        Dict with per-case results and aggregate metrics.
    """
    from agent.wismo_chain import build_wismo_chain
    from guardrails.output_validator import OutputValidator

    dataset = load_eval_dataset()
    chain = build_wismo_chain()
    validator = OutputValidator()

    per_case_results = []
    latencies_ms: list[float] = []

    relevance_hits = 0
    not_found_hits = 0
    not_found_total = 0
    guardrail_triggers = 0
    faithfulness_hits = 0

    print(f"\nRunning {len(dataset)} eval cases…\n")

    for case in dataset:
        case_id = case["id"]
        query = case["query"]
        tracking_id = case.get("tracking_id") or ""
        expected_status = case.get("expected_status", "")
        expected_contains = case.get("expected_answer_contains", [])

        # --- Run chain ---
        t0 = time.perf_counter()
        try:
            result = chain.invoke({"tracking_id": tracking_id, "query": query})
            answer = result.answer
            retrieved = result.tracking_data
            chain_error = None
        except Exception as exc:
            answer = f"ERROR: {exc}"
            retrieved = {}
            chain_error = str(exc)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

        # --- Relevance ---
        is_relevant = relevance_check(answer, expected_contains) if expected_contains else True
        if is_relevant:
            relevance_hits += 1

        # --- Not-found handling ---
        if expected_status == "NOT_FOUND":
            not_found_total += 1
            if not_found_check(answer):
                not_found_hits += 1

        # --- Guardrail ---
        validation = validator.validate(answer, retrieved)
        triggered = not validation.is_valid
        if triggered:
            guardrail_triggers += 1

        # --- Faithfulness (inverse of guardrail trigger) ---
        if not triggered:
            faithfulness_hits += 1

        per_case_results.append({
            "id": case_id,
            "query": query,
            "tracking_id": tracking_id,
            "expected_status": expected_status,
            "answer": answer,
            "is_relevant": is_relevant,
            "guardrail_triggered": triggered,
            "guardrail_violations": validation.violations,
            "latency_ms": round(elapsed_ms, 1),
            "error": chain_error,
        })

        status_icon = "✓" if is_relevant and not triggered else "✗"
        print(
            f"  [{status_icon}] Case {case_id:>2} | {expected_status:<20} | "
            f"{elapsed_ms:6.0f}ms | guardrail={'YES' if triggered else 'no '}"
        )

    # ---------------------------------------------------------------------------
    # Aggregate metrics
    # ---------------------------------------------------------------------------
    n = len(dataset)
    metrics = {
        "total_cases": n,
        "answer_faithfulness_pct": round(faithfulness_hits / n * 100, 1),
        "answer_relevance_pct": round(relevance_hits / n * 100, 1),
        "not_found_handling_pct": round(not_found_hits / not_found_total * 100, 1) if not_found_total else 0.0,
        "avg_latency_ms": round(sum(latencies_ms) / n, 1),
        "p95_latency_ms": round(sorted(latencies_ms)[int(n * 0.95)], 1),
        "guardrail_trigger_rate_pct": round(guardrail_triggers / n * 100, 1),
    }

    results = {
        "metrics": metrics,
        "per_case": per_case_results,
    }

    # Save
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    return results


def print_summary(results: dict[str, Any]) -> None:
    """Print a formatted summary table."""
    m = results["metrics"]
    print("\n" + "=" * 55)
    print("  WISMO AI ASSISTANT — EVALUATION RESULTS")
    print("=" * 55)
    print(f"  {'Metric':<35} {'Value':>8}")
    print("-" * 55)
    print(f"  {'Total cases':<35} {m['total_cases']:>8}")
    print(f"  {'Answer Faithfulness':<35} {m['answer_faithfulness_pct']:>7.1f}%")
    print(f"  {'Answer Relevance':<35} {m['answer_relevance_pct']:>7.1f}%")
    print(f"  {'Not-Found Handling':<35} {m['not_found_handling_pct']:>7.1f}%")
    print(f"  {'Avg Latency':<35} {m['avg_latency_ms']:>6.0f}ms")
    print(f"  {'p95 Latency':<35} {m['p95_latency_ms']:>6.0f}ms")
    print(f"  {'Guardrail Trigger Rate':<35} {m['guardrail_trigger_rate_pct']:>7.1f}%")
    print("=" * 55)
    print(f"\n  Full results saved to: evaluation/results.json\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_evals()
    print_summary(results)
