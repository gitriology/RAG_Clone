"""
Pipeline entry point for MS-ARC.
"""

from backend.ms_arc.state.retrieval_state import (
    RetrievalState
)

from backend.ms_arc.complexity.analyzer import (
    QueryComplexityAnalyzer
)

from backend.ms_arc.retrieval.retrieve import (
    retrieve
)

from backend.ms_arc.agreement.agreement import (
    compute_agreement
)

from backend.ms_arc.confidence.margin import (
    compute_margin
)

from backend.ms_arc.stability.stability import (
    compute_stability
)

from backend.ms_arc.decision.decision_engine import (
    compute_decision
)

from backend.ms_arc.retrieval.candidate_controller import (
    determine_candidate_k,
)


# ==========================================================
# MS-ARC PIPELINE
# ==========================================================

def run_msarc(
    query: str
) -> RetrievalState:

    state = RetrievalState(
        query=query
    )

    # ======================================================
    # PHASE 7 : QUERY COMPLEXITY
    # ======================================================

    analyzer = QueryComplexityAnalyzer()

    state = analyzer.analyze(
        state
    )

    print(
        "[Optimization #6] "
        f"Query complexity: "
        f"{state.query_complexity:.3f}"
    )

    print(
        "[Optimization #6] "
        f"Query type: "
        f"{state.query_type}"
    )

    # ======================================================
    # OPTIMIZATION #6
    # INITIAL CANDIDATE DEPTH
    # ======================================================
    #
    # At this stage:
    #
    #   agreement = unknown
    #   stability = unknown
    #
    # Therefore neutral values are used.
    #
    # After the first retrieval, actual agreement and
    # stability are measured and the controller is called
    # again.
    # ======================================================

    initial_candidate_k = (
        determine_candidate_k(

            query_complexity=
                state.query_complexity,

            query_type=
                state.query_type,

            agreement=0.5,

            stability=0.5,

            requested_k=
                state.recommended_topk,

            query=
                state.query,

        )
    )

    state.debug[
        "initial_candidate_k"
    ] = initial_candidate_k

    print(
        "[Optimization #6] "
        f"Selected initial candidate depth: "
        f"{initial_candidate_k}"
    )

    # ======================================================
    # INITIAL RETRIEVAL
    # ======================================================

    state = retrieve(

        state,

        candidate_k=
            initial_candidate_k,

        fusion_method=
            "minmax",

    )

    # ======================================================
    # PHASE 8 : RETRIEVAL SIGNALS
    # ======================================================

    state = compute_agreement(
        state
    )

    state = compute_margin(
        state
    )

    # ------------------------------------------------------
    # Stability performs ONE additional retrieval.
    # ------------------------------------------------------

    state = compute_stability(
        state
    )

    # ======================================================
    # ACTUAL SIGNALS
    # ======================================================

    actual_agreement = (
        state.signals.agreement.score
    )

    actual_stability = (
        state.signals.stability.score
    )

    print(
        "[Optimization #6] "
        f"Actual agreement: "
        f"{actual_agreement:.4f}"
    )

    print(
        "[Optimization #6] "
        f"Actual stability: "
        f"{actual_stability:.4f}"
    )

    # ======================================================
    # OPTIMIZATION #6
    # FINAL ADAPTIVE CANDIDATE DEPTH
    # ======================================================

    adaptive_candidate_k = (
        determine_candidate_k(

            query_complexity=
                state.query_complexity,

            query_type=
                state.query_type,

            agreement=
                actual_agreement,

            stability=
                actual_stability,

            requested_k=
                state.recommended_topk,

            query=
                state.query,

        )
    )

    state.debug[
        "adaptive_candidate_k"
    ] = adaptive_candidate_k

    print(
        "[Optimization #6] "
        f"Initial candidate pool: "
        f"{initial_candidate_k}"
    )

    print(
        "[Optimization #6] "
        f"Adaptive candidate pool "
        f"after signals: "
        f"{adaptive_candidate_k}"
    )

    # ======================================================
    # EXPANSION DECISION
    # ======================================================

    if adaptive_candidate_k > initial_candidate_k:

        print()

        print(
            "[Optimization #6] "
            "Expanding retrieval candidate pool"
        )

        print(
            "[Optimization #6] "
            f"{initial_candidate_k} -> "
            f"{adaptive_candidate_k}"
        )

        # --------------------------------------------------
        # Final retrieval with adaptive depth.
        # --------------------------------------------------

        state = retrieve(

            state,

            candidate_k=
                adaptive_candidate_k,

            fusion_method=
                "minmax",

        )

        # --------------------------------------------------
        # Recompute retrieval signals because the actual
        # retrieval pool changed.
        # --------------------------------------------------

        state = compute_agreement(
            state
        )

        state = compute_margin(
            state
        )

        print(
            "[Optimization #6] "
            "Final retrieval expanded successfully"
        )

    else:

        print(
            "[Optimization #6] "
            "No candidate expansion required"
        )

    # ======================================================
    # FINAL CANDIDATE DEPTH
    # ======================================================

    state.debug[
        "final_candidate_k"
    ] = adaptive_candidate_k

    state.debug[
        "candidate_expanded"
    ] = (
        adaptive_candidate_k
        >
        initial_candidate_k
    )

    state.debug[
        "candidate_depth_delta"
    ] = (
        adaptive_candidate_k
        -
        initial_candidate_k
    )

    # ======================================================
    # PHASE 9 : DECISION
    # ======================================================

    state = compute_decision(
        state
    )

    return state