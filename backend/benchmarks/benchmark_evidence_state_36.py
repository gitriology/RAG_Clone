"""Synthetic micro-benchmark for Optimization #36."""

from __future__ import annotations

import time

from backend.evidence_state.incremental import update_incremental_evidence_state
from backend.evidence_state.state.evidence_state import EvidenceState
from backend.ms_arc.state.retrieval_state import RetrievalState


def make_state():
    state = RetrievalState(query="What are the applications of satellite communication?")
    state.query_complexity = 0.35
    state.recommended_topk = 10
    state.signals.agreement.score = 0.82
    state.signals.margin.normalized_margin = 0.78
    state.signals.stability.score = 0.85
    state.signals.novelty.score = 0.70
    state.signals.evidence.score = 0.80
    state.signals.evidence.coverage = 0.75
    state.signals.evidence.diversity = 0.70
    state.signals.evidence.consistency = 0.80
    state.retrieval_confidence = 0.81
    return state


def benchmark(iterations=1000):
    retrieval = make_state()

    start = time.perf_counter()
    evidence = None
    for _ in range(iterations):
        evidence = update_incremental_evidence_state(retrieval, evidence)
    incremental = time.perf_counter() - start

    # A deliberately repeated construction approximates the avoidable work of
    # rebuilding the retrieval-stage state object each iteration. This benchmark
    # measures state-update overhead only; it is not an end-to-end RAG benchmark.
    start = time.perf_counter()
    for _ in range(iterations):
        update_incremental_evidence_state(retrieval, EvidenceState())
    rebuild = time.perf_counter() - start

    reduction = max(0.0, (rebuild - incremental) / rebuild * 100.0) if rebuild else 0.0

    print("\nOptimization #36 — incremental Evidence State benchmark\n")
    print(f"Iterations: {iterations}")
    print(f"Incremental reused-state time: {incremental:.6f}s")
    print(f"Fresh-state rebuild time: {rebuild:.6f}s")
    print(f"Measured update-overhead reduction: {reduction:.2f}%")
    print("Note: synthetic micro-benchmark; EvidenceState update overhead only, not end-to-end RAG latency.")


if __name__ == "__main__":
    benchmark()
