from backend.ms_arc.state.retrieval_state import RetrievalState
from backend.ms_arc.state.retrieval_signals import StabilityMetrics

from backend.retrieval.hybrid.hybrid_search import hybrid_search

from backend.ms_arc.retrieval.retrieve import (
    faiss_index,
    bm25,
    texts,
)


# ==========================================================
# JACCARD SIMILARITY
# ==========================================================

def jaccard(a, b):
    """
    Computes Jaccard similarity between two retrieval sets.
    """

    a = set(a)
    b = set(b)

    if not a and not b:
        return 1.0

    return len(a & b) / len(a | b)


# ==========================================================
# COMPUTE STABILITY
# ==========================================================

def compute_stability(
    state: RetrievalState,
) -> RetrievalState:
    """
    Measures retrieval stability by running
    retrieval with increasing Top-K values and
    comparing the retrieved document sets.
    """

    base_k = state.recommended_topk

    runs = []

    # =====================================================
    # Retrieval at K, K+2 and K+4
    # =====================================================

    for k in (
        base_k,
        base_k + 2,
        base_k + 4,
    ):

        results = hybrid_search(
            query=state.query,
            faiss_index=faiss_index,
            bm25=bm25,
            texts=texts,
            k=k,
        )

        merged = results["merged_results"]

        ids = [

            str(doc["doc_id"])

            for doc in merged

        ]

        runs.append(ids)

    # =====================================================
    # Jaccard Overlap
    # =====================================================

    overlap_1 = jaccard(
        runs[0],
        runs[1],
    )

    overlap_2 = jaccard(
        runs[1],
        runs[2],
    )

    stability_score = (

        overlap_1 +

        overlap_2

    ) / 2.0

    # =====================================================
    # Typed Metrics
    # =====================================================

    state.signals.stability = StabilityMetrics(

        score=stability_score,

        overlap_1=overlap_1,

        overlap_2=overlap_2,

        topk_runs=[
            base_k,
            base_k + 2,
            base_k + 4,
        ],

    )

    return state