from backend.evidence_state.state.evidence_state import (
    EvidenceFeature,
)


def extract_diagnostics_features(
    evidence_state,
):
    """
    Extract diagnostic features from an already-built
    EvidenceState.

    During the initial build, these values are initialized
    to zero. After diagnostics run, this extractor can be
    called again to refresh them.
    """

    signals = evidence_state.signals

    stats = evidence_state.statistics

    features = [

        EvidenceFeature(
            "health_score",
            signals.health.score,
            "diagnostics",
        ),

        EvidenceFeature(
            "uncertainty",
            signals.uncertainty.score,
            "diagnostics",
        ),

        EvidenceFeature(
            "vector_sparsity",
            stats.sparsity,
            "diagnostics",
        ),

        EvidenceFeature(
            "vector_norm",
            stats.l2_norm,
            "diagnostics",
        ),

    ]

    return features