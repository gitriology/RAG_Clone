from backend.ms_arc.state.retrieval_state import (
    RetrievalState,
)

from backend.ms_arc.state.retrieval_signals import (
    StabilityMetrics,
)

from backend.retrieval.hybrid.hybrid_search import (
    hybrid_search,
)

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
    Computes Jaccard similarity between two
    retrieval sets.
    """

    a = set(a)
    b = set(b)

    if not a and not b:
        return 1.0

    union = a | b

    if not union:
        return 1.0

    return len(
        a & b
    ) / len(union)


# ==========================================================
# COMPUTE STABILITY
# ==========================================================

def compute_stability(
    state: RetrievalState,
) -> RetrievalState:
    """
    Measures retrieval stability using increasing
    Top-K values.

    IMPORTANT:
    The same fusion method used by the main retrieval
    is used here as well.

    This prevents an RRF benchmark run from silently
    using Min-Max during stability evaluation.
    """

    base_k = (
        state.recommended_topk
    )

    fusion_method = (
        state.debug.get(
            "fusion_method",
            "minmax",
        )
    )

    # ======================================================
    # Top-K runs
    # ======================================================

    topk_values = [

        base_k,

        base_k + 2,

        base_k + 4,

    ]

    runs = []

    # ======================================================
    # Retrieval
    # ======================================================

    for k in topk_values:

        results = hybrid_search(

            query=state.query,

            faiss_index=faiss_index,

            bm25=bm25,

            texts=texts,

            k=k,

            fusion_method=fusion_method,

        )

        merged = results[
            "merged_results"
        ]

        ids = [

            str(
                doc["doc_id"]
            )

            for doc in merged

        ]

        runs.append(ids)

    # ======================================================
    # Jaccard Overlap
    # ======================================================

    overlap_1 = jaccard(

        runs[0],

        runs[1],

    )

    overlap_2 = jaccard(

        runs[1],

        runs[2],

    )

    stability_score = (

        overlap_1
        +
        overlap_2

    ) / 2.0

    # ======================================================
    # Typed Metrics
    # ======================================================

    state.signals.stability = (
        StabilityMetrics(

            score=stability_score,

            overlap_1=overlap_1,

            overlap_2=overlap_2,

            topk_runs=topk_values,

        )
    )

    # ======================================================
    # Debug
    # ======================================================

    state.debug[
        "stability_fusion_method"
    ] = fusion_method

    state.debug[
        "stability_topk_runs"
    ] = topk_values

    state.debug[
        "stability_overlap_1"
    ] = overlap_1

    state.debug[
        "stability_overlap_2"
    ] = overlap_2

    print(
        "[Optimization #5 + #6] "
        "Stability fusion:",
        fusion_method,
    )

    return state