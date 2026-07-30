from backend.evidence_state.state.evidence_state import (
    EvidenceFeature,
)


def extract_quality_features(
    graph_state,
):
    """
    Quality features derived from Phase 8.
    """

    quality = graph_state.signals.quality

    ranking = graph_state.signals.ranking

    features = [

        EvidenceFeature(
            "evidence_quality",
            quality.score,
            "quality",
        ),

        EvidenceFeature(
            "graph_consensus",
            ranking.graph_consensus,
            "quality",
        ),

    ]

    return features