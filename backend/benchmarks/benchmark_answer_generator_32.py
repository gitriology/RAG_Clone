"""Micro-benchmark for Optimization #32.

Compares the number of embedding inference calls required by the pre-#32
query-then-candidates flow with the optimized single batched call.
The benchmark uses a deterministic fake encoder so results measure call
reduction rather than hardware/model latency.
"""

from __future__ import annotations

import time


class FakeEncoder:
    def __init__(self):
        self.calls = 0
        self.items = 0

    def encode(self, texts):
        self.calls += 1
        self.items += len(texts)
        # Small deterministic workload.
        return [sum(ord(c) for c in str(text)) % 997 for text in texts]


def baseline(encoder, query, sentences):
    encoder.encode([query])
    encoder.encode(sentences)


def optimized(encoder, query, sentences):
    encoder.encode([query, *sentences])


def run():
    queries = [
        "What is FAISS?",
        "What is hybrid retrieval?",
        "Explain BM25 ranking.",
        "What is evidence fusion?",
        "How does adaptive retrieval work?",
    ]
    sentences = [
        "FAISS is a library for efficient similarity search.",
        "BM25 is a lexical retrieval method.",
        "Hybrid retrieval combines dense and sparse signals.",
        "Evidence fusion combines selected evidence sentences.",
        "Adaptive retrieval can stop when evidence is sufficient.",
        "Cross-encoder scores can rank retrieved documents.",
    ]

    baseline_encoder = FakeEncoder()
    optimized_encoder = FakeEncoder()

    start = time.perf_counter()
    for query in queries:
        baseline(baseline_encoder, query, sentences)
    baseline_time = time.perf_counter() - start

    start = time.perf_counter()
    for query in queries:
        optimized(optimized_encoder, query, sentences)
    optimized_time = time.perf_counter() - start

    print("Optimization #32: Answer-generator embedding batching")
    print(f"Baseline encode calls   : {baseline_encoder.calls}")
    print(f"Optimized encode calls  : {optimized_encoder.calls}")
    print(f"Calls removed           : {baseline_encoder.calls - optimized_encoder.calls}")
    print(f"Baseline encoded items  : {baseline_encoder.items}")
    print(f"Optimized encoded items : {optimized_encoder.items}")
    print(f"Baseline fake time      : {baseline_time:.6f}s")
    print(f"Optimized fake time     : {optimized_time:.6f}s")
    print("Note: fake timing is illustrative; call-count reduction is the measured optimization.")


if __name__ == "__main__":
    run()
