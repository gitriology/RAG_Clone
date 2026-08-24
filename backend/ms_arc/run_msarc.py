"""
Pipeline entry point for MS-ARC.
"""

from backend.ms_arc.state.retrieval_state import (
    RetrievalState,
)

from backend.ms_arc.complexity.analyzer import (
    QueryComplexityAnalyzer,
)

from backend.ms_arc.retrieval.retrieve import (
    retrieve,
)

from backend.ms_arc.agreement.agreement import (
    compute_agreement,
)

from backend.ms_arc.confidence.margin import (
    compute_margin,
)

from backend.ms_arc.stability.stability import (
    compute_stability,
)

from backend.ms_arc.decision.decision_engine import (
    compute_decision,
)


def run_msarc(
    query: str,
    fusion_method: str = "minmax",
) -> RetrievalState:
    """
    Execute the complete MS-ARC pipeline.

    fusion_method is explicitly propagated through
    the retrieval pipeline.

    Supported:

        minmax
        rrf
    """

    fusion_method = (
        fusion_method
        or "minmax"
    ).lower().strip()

    if fusion_method not in {
        "minmax",
        "rrf",
    }:

        raise ValueError(
            f"Unsupported fusion method: "
            f"{fusion_method}"
        )

    state = RetrievalState(
        query=query
    )

    # ======================================================
    # Store experiment configuration
    # ======================================================

    state.debug[
        "fusion_method"
    ] = fusion_method

    # ======================================================
    # Phase 7 — Query Complexity
    # ======================================================

    analyzer = (
        QueryComplexityAnalyzer()
    )

    state = analyzer.analyze(
        state
    )

    # ======================================================
    # Retrieval
    # ======================================================

    state = retrieve(

        state,

        fusion_method=fusion_method,

    )

    # ======================================================
    # Phase 8 — Retrieval Signals
    # ======================================================

    state = compute_agreement(
        state
    )

    state = compute_margin(
        state
    )

    state = compute_stability(
        state
    )

    # ======================================================
    # Phase 9 — Decision
    # ======================================================

    state = compute_decision(
        state
    )

    return state