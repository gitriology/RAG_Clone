from backend.evidence_state.state.evidence_state import (
    EvidenceState,
)

from backend.evidence_state.features.retrieval_features import (
    extract_retrieval_features,
)

from backend.evidence_state.features.graph_features import (
    extract_graph_features,
)

from backend.evidence_state.features.quality_features import (
    extract_quality_features,
)

from backend.evidence_state.features.ranking_features import (
    extract_ranking_features,
)

from backend.evidence_state.features.diagnostics_features import (
    extract_diagnostics_features,
)


# ==========================================================
# BUILD EVIDENCE FEATURES
# ==========================================================

def build_features(
    retrieval_state,
    graph_state,
) -> EvidenceState:
    """
    Builds the complete Phase-9 Evidence Feature Set.

    Pipeline

        1. Retrieval Features
        2. Graph Features
        3. Quality Features
        4. Ranking Features
        5. Diagnostics Features
    """

    evidence_state = EvidenceState()

    # ======================================================
    # Retrieval Features
    # ======================================================

    evidence_state.features.extend(

        extract_retrieval_features(
            retrieval_state
        )

    )

    # ======================================================
    # Graph Features
    # ======================================================

    evidence_state.features.extend(

        extract_graph_features(
            graph_state
        )

    )

    # ======================================================
    # Quality Features
    # ======================================================

    evidence_state.features.extend(

        extract_quality_features(
            graph_state
        )

    )

    # ======================================================
    # Ranking Features
    # ======================================================

    evidence_state.features.extend(

        extract_ranking_features(
            graph_state
        )

    )


    # ======================================================
    # Build Lookup Table
    # ======================================================

    evidence_state.feature_lookup = {

        feature.name: feature

        for feature in evidence_state.features

    }

    return evidence_state