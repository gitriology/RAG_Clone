from backend.evidence_state.state.evidence_state import (
    EvidenceFeature,
)


def extract_graph_features(
    graph_state,
):
    """
    Phase 8 → Graph Features
    """

    graph = graph_state.signals

    centrality = graph.centrality

    features = [

        EvidenceFeature(
            "graph_score",
            graph_state.graph_score,
            "graph",
        ),

        EvidenceFeature(
            "graph_density",
            graph.graph.graph_density,
            "graph",
        ),

        EvidenceFeature(
            "graph_coherence",
            graph.coherence.score,
            "graph",
        ),

        EvidenceFeature(
            "connected_components",
            float(graph.graph.connected_components),
            "graph",
        ),

        EvidenceFeature(
            "average_degree",
            centrality.average_degree,
            "graph",
        ),

        EvidenceFeature(
            "average_betweenness",
            centrality.average_betweenness,
            "graph",
        ),

        EvidenceFeature(
            "average_closeness",
            centrality.average_closeness,
            "graph",
        ),

        EvidenceFeature(
            "average_eigenvector",
            centrality.average_eigenvector,
            "graph",
        ),

        EvidenceFeature(
            "average_pagerank",
            centrality.average_pagerank,
            "graph",
        ),

    ]

    return features