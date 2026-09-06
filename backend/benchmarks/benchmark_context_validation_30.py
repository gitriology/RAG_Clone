"""Optimization #30 benchmark: validation embedding reuse.

This benchmark isolates embedding-call behavior with a deterministic fake
MiniLM-like encoder. It measures model.encode invocation count, not real model
latency, so it remains dependency-light and reproducible.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from backend.generation.validation.embedding_cache import ValidationEmbeddingCache


class FakeMiniLM:
    def __init__(self):
        self.calls = 0
        self.items = 0

    def encode(self, texts, **kwargs):
        values = [str(x) for x in texts]
        self.calls += 1
        self.items += len(values)
        return np.asarray(
            [[float((sum(map(ord, text)) % 997) + i + 1) for i in range(8)] for text in values],
            dtype=np.float32,
        )


def run_baseline():
    model = FakeMiniLM()
    start = perf_counter()
    query = "What is FAISS?"
    doc = "FAISS is a library for efficient similarity search."
    answer = doc

    # Mirrors the pre-#30 behavior: every stage encodes independently.
    model.encode([query], normalize_embeddings=True)
    model.encode([doc], normalize_embeddings=True)
    model.encode([doc], normalize_embeddings=True)  # sentence candidates
    model.encode([answer], normalize_embeddings=True)
    model.encode([doc], normalize_embeddings=True)  # selected evidence
    model.encode([query], normalize_embeddings=True)

    return {
        "encode_calls": model.calls,
        "encoded_items": model.items,
        "elapsed_ms": (perf_counter() - start) * 1000,
    }


def run_optimized():
    model = FakeMiniLM()
    cache = ValidationEmbeddingCache(model=model)
    start = perf_counter()
    query = "What is FAISS?"
    doc = "FAISS is a library for efficient similarity search."
    answer = doc

    cache.encode([query])
    cache.encode([doc])
    cache.encode([doc])       # reused document sentence
    cache.encode([answer])
    cache.encode([doc])       # reused selected evidence
    cache.encode([query])     # reused query

    info = cache.info()
    return {
        "encode_calls": info["encode_calls"],
        "encoded_items": info["encoded_items"],
        "reused_items": info["reused_items"],
        "elapsed_ms": (perf_counter() - start) * 1000,
    }


def main():
    baseline = run_baseline()
    optimized = run_optimized()
    removed = baseline["encode_calls"] - optimized["encode_calls"]

    print("Optimization #30 — Context Validation embedding reuse benchmark")
    print()
    print("Baseline encode calls:", baseline["encode_calls"])
    print("Optimized encode calls:", optimized["encode_calls"])
    print("Validation embedding encode calls removed:", removed)
    print("Baseline encoded items:", baseline["encoded_items"])
    print("Optimized encoded items:", optimized["encoded_items"])
    print("Optimized reused items:", optimized["reused_items"])
    print(f"Baseline measured fake-encoding time: {baseline['elapsed_ms']:.4f} ms")
    print(f"Optimized measured fake-encoding time: {optimized['elapsed_ms']:.4f} ms")
    print()
    print("Note: timing uses a fake encoder; use real MiniLM profiling for latency claims.")


if __name__ == "__main__":
    main()
