from __future__ import annotations

from typing import Any

from backend.evidence_state.feature_builder import (
    build_features,
)

from backend.evidence_state.feature_groups import (
    build_feature_groups,
)

from backend.evidence_state.normalization import (
    normalize_features,
)

from backend.evidence_state.vector_builder import (
    build_vector,
    update_weighted_statistics,
)

from backend.evidence_state.weighting.feature_weights import (
    apply_feature_weights,
)

from backend.evidence_state.weighting.feature_importance import (
    compute_feature_importance,
)

from backend.evidence_state.weighting.weighted_vector import (
    build_weighted_vector,
)

from backend.evidence_state.weighting.evidence_score import (
    compute_evidence_score,
)

from backend.evidence_state.diagnostics.profile import (
    build_profile,
)

from backend.evidence_state.diagnostics.feature_diagnostics import (
    diagnose_features,
)

from backend.evidence_state.diagnostics.vector_health import (
    compute_vector_health,
)

from backend.evidence_state.diagnostics.uncertainty import (
    compute_uncertainty,
)

from backend.evidence_state.reasoning.reasoning_builder import (
    build_reasoning_state,
)

from backend.ms_arc.state.retrieval_state import (
    RetrievalState,
)

from backend.evidence_graph.state.evidence_graph_state import (
    EvidenceGraphState,
)


# ==========================================================
# BUILD COMPLETE EVIDENCE STATE
# ==========================================================

def build_evidence_state(
    retrieval_state: RetrievalState,
    graph_state: EvidenceGraphState,
) -> Any:
    """
    Phase 9
    Complete Evidence State pipeline.

    Pipeline
        Stage 1 : Feature Extraction
        Stage 2 : Feature Organization
        Stage 3 : Normalization
        Stage 4 : Feature Weighting
        Stage 5 : Vector Construction
        Stage 6 : Diagnostics
        Stage 7 : Reasoning State

    IMPORTANT
    ---------
    EvidenceState requires BOTH:

        retrieval_state
        graph_state

    RetrievalState provides the original retrieval,
    ranking and MS-ARC information.

    EvidenceGraphState provides the graph, nodes,
    edges, clusters, graph statistics and graph score.
    """

    if retrieval_state is None:
        raise ValueError(
            "build_evidence_state() received "
            "retrieval_state=None."
        )

    if graph_state is None:
        raise ValueError(
            "build_evidence_state() received "
            "graph_state=None."
        )

    # ======================================================
    # Stage 1
    # Feature Extraction
    # ======================================================

    state = build_features(
        retrieval_state,
        graph_state,
    )

    # ======================================================
    # Stage 2
    # Feature Groups
    # ======================================================

    state = build_feature_groups(
        state,
    )

    # ======================================================
    # Stage 3
    # Normalization
    # ======================================================

    state = normalize_features(
        state,
    )

    # ======================================================
    # Stage 4
    # Feature Weighting
    # ======================================================

    state = apply_feature_weights(
        state,
    )

    state = compute_feature_importance(
        state,
    )

    # ======================================================
    # Stage 5
    # Vector Construction
    # ======================================================

    state = build_vector(
        state,
    )

    state = build_weighted_vector(
        state,
    )

    state = update_weighted_statistics(
        state,
    )

    state = compute_evidence_score(
        state,
    )

    # ======================================================
    # Stage 6
    # Diagnostics
    # ======================================================

    state = build_profile(
        state,
    )

    state = diagnose_features(
        state,
    )

    state = compute_vector_health(
        state,
    )

    state = compute_uncertainty(
        state,
    )

    # ======================================================
    # Stage 7
    # Reasoning State
    # ======================================================

    state = build_reasoning_state(
        retrieval_state,
        graph_state,
        state,
    )

    return state