from backend.evidence_graph.state.evidence_graph_state import (
    EvidenceGraphState,
)


def compute_graph_score(
    graph_state: EvidenceGraphState,
) -> float:
    """
    Computes the overall quality score of the
    Evidence Graph.

    Components

        • Graph Density
        • Connectivity
        • Graph Coherence
        • Evidence Quality
        • Average Node Importance
    """

    stats = graph_state.statistics

    density = stats.graph_density

    connectivity = (

        1.0 /

        max(
            stats.connected_components,
            1,
        )

    )

    coherence = (

        graph_state.signals.coherence.score

    )

    evidence_quality = (

        graph_state.signals.quality.score

    )

    importance = (

        graph_state.signals
        .importance
        .average_importance

    )

    score = (

        0.15 * density +

        0.15 * connectivity +

        0.25 * coherence +

        0.20 * evidence_quality +

        0.25 * importance

    )

    graph_state.graph_score = round(

        min(score, 1.0),

        4,

    )

    return graph_state.graph_score