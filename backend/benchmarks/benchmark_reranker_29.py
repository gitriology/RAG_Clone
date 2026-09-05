"""Micro-benchmark for Optimization #29.

This benchmark isolates Cross-Encoder invocation count rather than loading a
large model.  It demonstrates the computational work removed by reusing the
canonical MS-ARC scores downstream.
"""

from __future__ import annotations

import time

from backend.ms_arc.state.retrieval_state import RetrievedDocument
from backend.reranker.model.reranker import rerank


class FakeCrossEncoder:
    def __init__(self):
        self.predict_calls = 0

    def predict(self, pairs):
        self.predict_calls += 1
        return [float(len(pair[1])) for pair in pairs]


def main():
    docs = [
        RetrievedDocument(str(i), f"document {i} with evidence")
        for i in range(5)
    ]
    pairs = [("query", doc.text) for doc in docs]

    baseline_model = FakeCrossEncoder()
    t0 = time.perf_counter()
    baseline_model.predict(pairs)  # MS-ARC pass
    baseline_model.predict(pairs)  # duplicate downstream pass
    baseline_time = time.perf_counter() - t0

    optimized_model = FakeCrossEncoder()
    scores = optimized_model.predict(pairs)  # canonical MS-ARC pass
    for doc, score in zip(docs, scores):
        doc.rerank_score = score

    t1 = time.perf_counter()
    rerank("query", docs)  # reuse only; no predict
    optimized_time = time.perf_counter() - t1

    print("Optimization #29 — Cross-Encoder duplicate-inference benchmark")
    print(f"Documents: {len(docs)}")
    print(f"Baseline predict calls: {baseline_model.predict_calls}")
    print(f"Optimized predict calls: {optimized_model.predict_calls}")
    print(f"Baseline measured scoring time: {baseline_time * 1000:.4f} ms")
    print(f"Optimized downstream rerank time: {optimized_time * 1000:.4f} ms")
    print("Cross-Encoder inference passes removed: 1")


if __name__ == "__main__":
    main()
