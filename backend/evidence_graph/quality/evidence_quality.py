from backend.evidence_graph.state.evidence_graph_state import (
    EvidenceGraphState,
)

from backend.ms_arc.state.retrieval_state import (
    RetrievalState,
)


def compute_evidence_quality(
    retrieval_state: RetrievalState,
    graph_state: EvidenceGraphState,
) -> EvidenceGraphState:
    """
    Computes the overall evidence quality.

    Combines:

        • Retrieval confidence
        • Graph score
        • Graph coherence

    Output:

        graph_state.signals.quality
    """

    # =====================================================
    # Retrieve inputs
    # =====================================================

    retrieval_confidence = (
        retrieval_state.signals.decision.confidence
    )

    graph_score = (
        graph_state.graph_score
    )

    coherence_score = (
        graph_state.signals.coherence.score
    )

    # =====================================================
    # Weighted Quality Score
    # =====================================================

    quality_score = (

        0.40 * retrieval_confidence +

        0.30 * graph_score +

        0.30 * coherence_score

    )

    quality_score = max(
        0.0,
        min(quality_score, 1.0)
    )

    # =====================================================
    # Store typed metrics
    # =====================================================

    graph_state.signals.quality.score = (
        quality_score
    )

    graph_state.signals.quality.graph_score = (
        graph_score
    )

    graph_state.signals.quality.coherence_score = (
        coherence_score
    )

    graph_state.signals.quality.retrieval_confidence = (
        retrieval_confidence
    )

    return graph_state