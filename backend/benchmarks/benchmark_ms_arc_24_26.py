"""Benchmark MS-ARC Optimizations #24-#26.

#24 compares dependency-light complexity analysis with the optional spaCy mode.
#25 verifies that the same four complexity signals remain available.
#26 reports the adaptive retrieval plan produced for representative queries.
"""

from __future__ import annotations

import time

from backend.ms_arc.complexity.analyzer import QueryComplexityAnalyzer
from backend.ms_arc.retrieval.adaptive_policy import AdaptiveRetrievalPolicy
from backend.ms_arc.state.retrieval_state import RetrievalState


QUERIES = [
    "What is FAISS?",
    "What is BM25?",
    "How does FAISS compare with BM25?",
    "Explain the relationship between retrieval latency and precision in a RAG pipeline.",
    "Compare FAISS and BM25 ranking stability, precision, recall, and latency.",
]


def benchmark_complexity(mode: str) -> None:
    analyzer = QueryComplexityAnalyzer(ner_mode=mode)
    start = time.perf_counter()
    states = [analyzer.analyze(RetrievalState(query=q)) for q in QUERIES]
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"{mode:>12}: {elapsed_ms:8.3f} ms total | {elapsed_ms / len(QUERIES):.3f} ms/query")
    print("  signals:", [
        states[0].debug["complexity"]["query_length"],
        states[0].debug["complexity"]["named_entities"],
        states[0].debug["complexity"]["technical_score"],
        states[0].debug["complexity"]["multihop_score"],
    ])


def main() -> None:
    print("=" * 68)
    print("OPTIMIZATIONS #24-#26 - MS-ARC BENCHMARK")
    print("=" * 68)
    print(f"Queries: {len(QUERIES)}")
    print()

    benchmark_complexity("lightweight")
    try:
        benchmark_complexity("spacy")
    except Exception as exc:
        print(f"       spacy: unavailable ({type(exc).__name__}: {exc})")

    print()
    policy = AdaptiveRetrievalPolicy()
    for query in QUERIES:
        state = QueryComplexityAnalyzer().analyze(RetrievalState(query=query))
        plan = policy.plan(state.query_complexity, state.recommended_topk)
        print(
            f"{query}\n"
            f"  complexity={state.query_complexity:.3f} type={state.query_type} "
            f"recommended_topk={state.recommended_topk}\n"
            f"  plan: initial_k={plan.initial_k}, +{plan.expansion_step}, max_k={plan.max_k}"
        )


if __name__ == "__main__":
    main()
