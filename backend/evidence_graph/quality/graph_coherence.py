import networkx as nx

from backend.evidence_graph.state.evidence_graph_state import (
    EvidenceGraphState,
)


def compute_graph_coherence(
    graph_state: EvidenceGraphState,
) -> EvidenceGraphState:
    """
    Computes overall graph coherence.

    Current formulation

        coherence =
            40% average edge weight
          + 30% connectivity
          + 30% cluster cohesion

    Output

        graph_state.signals.coherence
    """

    graph = graph_state.graph

    # =====================================================
    # Empty graph
    # =====================================================

    if graph.number_of_nodes() == 0:

        return graph_state

    # =====================================================
    # Average Edge Weight
    # =====================================================

    if graph.number_of_edges() == 0:

        average_edge_weight = 0.0

    else:

        weights = [

            data.get("weight", 0.0)

            for _, _, data in graph.edges(data=True)

        ]

        average_edge_weight = sum(weights) / len(weights)

    # =====================================================
    # Connectivity
    # =====================================================

    if graph.number_of_nodes() <= 1:

        connectivity = 1.0

    else:

        components = nx.number_connected_components(graph)

        connectivity = 1.0 / components

    # =====================================================
    # Cluster Cohesion
    # =====================================================

    if not graph_state.clusters:

        cluster_cohesion = 0.0

    else:

        cluster_cohesion = sum(

            cluster.score

            for cluster in graph_state.clusters

        ) / len(graph_state.clusters)

    # =====================================================
    # Final Coherence
    # =====================================================

    coherence = (

        0.40 * average_edge_weight +

        0.30 * connectivity +

        0.30 * cluster_cohesion

    )

    coherence = max(

        0.0,

        min(coherence, 1.0)

    )

    # =====================================================
    # Store typed metrics
    # =====================================================

    graph_state.signals.coherence.average_edge_weight = (
        average_edge_weight
    )

    graph_state.signals.coherence.connectivity = (
        connectivity
    )

    graph_state.signals.coherence.cluster_cohesion = (
        cluster_cohesion
    )

    graph_state.signals.coherence.score = (
        coherence
    )

    return graph_state