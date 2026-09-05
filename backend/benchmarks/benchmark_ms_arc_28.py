"""Benchmark Optimization #28: adaptive MS-ARC retrieval depth."""

from __future__ import annotations

import statistics
import time

from backend.ms_arc.run_msarc import run_msarc


QUERIES = [
    "What is FAISS?",
    "What is BM25?",
    "How does FAISS compare with BM25?",
    "Explain the relationship between retrieval latency and precision in a RAG pipeline.",
    "Compare FAISS and BM25 ranking stability, precision, recall, and latency.",
]


def main():
    print("=" * 68)
    print("OPTIMIZATION #28 - MS-ARC ADAPTIVE RETRIEVAL BENCHMARK")
    print("=" * 68)
    print(f"Queries: {len(QUERIES)}")
    print()

    rows = []
    for query in QUERIES:
        started = time.perf_counter()
        state = run_msarc(query)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        rows.append((query, state, elapsed_ms))

        print(f"Query: {query}")
        print(
            f"  complexity={state.query_complexity:.3f} "
            f"type={state.query_type}"
        )
        print(
            f"  plan: {state.debug['adaptive_plan']}"
        )
        print(
            f"  iterations={state.debug['adaptive_iterations_count']} "
            f"final_k={state.debug['adaptive_final_k']} "
            f"stop={state.debug['adaptive_stop_reason']}"
        )
        print(
            f"  confidence={state.retrieval_confidence:.4f} "
            f"latency_ms={elapsed_ms:.2f}"
        )
        print()

    final_ks = [state.debug["adaptive_final_k"] for _, state, _ in rows]
    iterations = [state.debug["adaptive_iterations_count"] for _, state, _ in rows]
    latencies = [elapsed for _, _, elapsed in rows]

    print("SUMMARY")
    print(f"  average final K: {statistics.mean(final_ks):.2f}")
    print(f"  average iterations: {statistics.mean(iterations):.2f}")
    print(f"  average latency ms: {statistics.mean(latencies):.2f}")
    print(f"  K distribution: {final_ks}")
    print()
    print("This benchmark measures adaptive depth; compare it against a fixed-K")
    print("baseline before claiming latency or compute savings in the paper.")


if __name__ == "__main__":
    main()
