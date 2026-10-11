from backend.ms_arc.state.retrieval_state import (
    RetrievalState,
)

from backend.ms_arc.state.retrieval_signals import (
    StabilityMetrics,
)
from backend.ms_arc.stability.metrics import jaccard, prefix_jaccard

from backend.retrieval.hybrid.hybrid_search import (
    hybrid_search,
)

from backend.ms_arc.retrieval.retrieve import (
    faiss_index,
    bm25,
    texts,
)


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

    # Optimization #28: when the adaptive loop has already produced
    # successive rankings, stability is computed from those rankings.
    # This avoids the old three extra FAISS/BM25 searches per iteration.
    adaptive_rankings = state.debug.get("adaptive_rankings", [])
    if adaptive_rankings:
        current = [str(x) for x in adaptive_rankings[-1]]
        previous = [str(x) for x in adaptive_rankings[-2]] if len(adaptive_rankings) > 1 else None

        if previous is None:
            # A single ranking cannot establish stability; use the existing
            # neutral prior rather than claiming either stability or instability.
            overlap_1 = 0.5
            overlap_2 = 0.5
            stability_score = 0.5
        else:
            # Adaptive retrieval increases K. Compare the previous ranking with
            # the equally sized prefix of the expanded ranking; otherwise a
            # perfectly preserved top-5 would score only 5/10 by Jaccard.
            overlap_1 = prefix_jaccard(previous, current)
            overlap_2 = overlap_1
            stability_score = overlap_1
        current_k = state.debug.get("retrieval_k_used", len(current))

        state.signals.stability = StabilityMetrics(
            score=stability_score,
            overlap_1=overlap_1,
            overlap_2=overlap_2,
            topk_runs=[current_k],
        )

        state.debug["stability_mode"] = "adaptive_rankings"
        state.debug["stability_topk_runs"] = [current_k]
        state.debug["stability_overlap_1"] = overlap_1
        state.debug["stability_overlap_2"] = overlap_2
        return state

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

    # Compare equal-sized top prefixes across increasing K values. Raw
    # Jaccard on full sets penalizes the larger result set even when the
    # original top-ranked documents remain unchanged.
    overlap_1 = prefix_jaccard(runs[0], runs[1])

    overlap_2 = prefix_jaccard(runs[1], runs[2])

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