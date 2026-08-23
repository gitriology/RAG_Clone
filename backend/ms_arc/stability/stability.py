from backend.ms_arc.state.retrieval_state import (
    RetrievalState
)

from backend.ms_arc.state.retrieval_signals import (
    StabilityMetrics
)

from backend.retrieval.hybrid.hybrid_search import (
    hybrid_search
)

from backend.ms_arc.retrieval.retrieve import (
    faiss_index,
    bm25,
    texts,
)

from backend.ms_arc.retrieval.candidate_controller import (
    determine_candidate_k,
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

    if not a or not b:

        return 0.0

    return len(a & b) / len(a | b)


# ==========================================================
# COMPUTE STABILITY
# ==========================================================

def compute_stability(
    state: RetrievalState,
) -> RetrievalState:

    """
    Computes retrieval stability using nested Top-K subsets.

    Optimization #5
    ----------------
    Only ONE additional hybrid retrieval is executed.

    Optimization #6
    ----------------
    Candidate depth for the stability retrieval is selected
    using:

        query complexity
        +
        query type
        +
        actual agreement

    Stability itself is intentionally neutral during this
    first stability retrieval because it is not available
    yet.
    """

    base_k = int(
        state.recommended_topk
    )

    max_k = base_k + 4

    # ======================================================
    # Agreement
    # ======================================================

    agreement = (
        state.signals.agreement.score
    )

    # ======================================================
    # Adaptive Candidate Pool
    # ======================================================

    candidate_k = determine_candidate_k(

        query_complexity=
            state.query_complexity,

        query_type=
            state.query_type,

        agreement=
            agreement,

        # Stability is not yet known.
        stability=0.5,

        requested_k=
            max_k,

        query=
            state.query,

    )

    print(
        "[Optimization #5 + #6] "
        "Stability retrieval"
    )

    print(
        f"Base Top-K          : "
        f"{base_k}"
    )

    print(
        f"Maximum Top-K       : "
        f"{max_k}"
    )

    print(
        f"Agreement           : "
        f"{agreement:.4f}"
    )

    print(
        f"Candidate pool      : "
        f"{candidate_k}"
    )

    # ======================================================
    # SINGLE ADDITIONAL RETRIEVAL
    # ======================================================

    results = hybrid_search(

        query=state.query,

        faiss_index=faiss_index,

        bm25=bm25,

        texts=texts,

        k=max_k,

        candidate_k=candidate_k,

        # Keep the project's baseline fusion.
        fusion_method="minmax",

    )

    merged = results[
        "merged_results"
    ]

    print(
        "[Optimization #5] "
        "Retrieved candidates for "
        f"stability: {len(merged)}"
    )

    # ======================================================
    # BUILD NESTED TOP-K RUNS
    # ======================================================

    topk_values = (

        base_k,

        base_k + 2,

        base_k + 4,

    )

    runs = []

    for k in topk_values:

        subset = merged[
            :k
        ]

        ids = [

            str(
                doc["doc_id"]
            )

            for doc in subset

        ]

        runs.append(
            ids
        )

        print(
            "[Optimization #5] "
            f"Derived Top-{k} "
            "from existing retrieval"
        )

    # ======================================================
    # JACCARD
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
    # DEBUG
    # ======================================================

    print(
        "[Optimization #5] "
        f"Top-K runs: "
        f"{list(topk_values)}"
    )

    print(
        "[Optimization #5] "
        f"Overlap K/K+2: "
        f"{overlap_1:.4f}"
    )

    print(
        "[Optimization #5] "
        f"Overlap K+2/K+4: "
        f"{overlap_2:.4f}"
    )

    print(
        "[Optimization #5] "
        f"Stability Score: "
        f"{stability_score:.4f}"
    )

    # ======================================================
    # STORE DEBUG INFORMATION
    # ======================================================

    state.debug[
        "stability_candidate_k"
    ] = candidate_k

    state.debug[
        "stability_merged_results"
    ] = merged

    state.debug[
        "stability_fusion_method"
    ] = results[
        "fusion_method"
    ]

    # ======================================================
    # TYPED METRICS
    # ======================================================

    state.signals.stability = (
        StabilityMetrics(

            score=stability_score,

            overlap_1=overlap_1,

            overlap_2=overlap_2,

            topk_runs=list(
                topk_values
            ),

        )
    )

    return state