import networkx as nx

from backend.evidence_graph.analytics.centrality import (
    compute_degree,
    compute_betweenness,
    compute_closeness,
    compute_pagerank,
)


# ==========================================================
# Eigenvector Centrality
# ==========================================================

def compute_eigenvector(graph: nx.Graph):
    """
    Computes Eigenvector Centrality.

    Falls back to zeros if convergence fails.
    """

    if graph.number_of_nodes() == 0:
        return {}

    try:

        return nx.eigenvector_centrality(
            graph,
            max_iter=1000,
            weight="weight",
        )

    except Exception:

        return {
            node: 0.0
            for node in graph.nodes
        }


# ==========================================================
# Importance Score
# ==========================================================

def compute_importance(
    node,
):
    """
    Overall node importance.

    Weighted combination of structural
    graph metrics.

    Sum(weights)=1.0
    """

    return (

        0.20 * node.degree_centrality +

        0.20 * node.betweenness_centrality +

        0.20 * node.closeness_centrality +

        0.20 * node.eigenvector_centrality +

        0.20 * node.pagerank

    )


# ==========================================================
# Graph Analysis
# ==========================================================

def analyze_graph(
    graph_state,
):
    """
    Computes graph analytics for every node.

    Populates

        • Degree

        • Betweenness

        • Closeness

        • Eigenvector

        • PageRank

        • Importance

    and updates GraphSignals.
    """

    graph = graph_state.graph

    if graph.number_of_nodes() == 0:
        return graph_state

    # ------------------------------------------------------
    # Centralities
    # ------------------------------------------------------

    degree = compute_degree(graph)

    betweenness = compute_betweenness(graph)

    closeness = compute_closeness(graph)

    pagerank = compute_pagerank(graph)

    eigenvector = compute_eigenvector(graph)

    # ------------------------------------------------------
    # Store on nodes
    # ------------------------------------------------------

    for node in graph_state.nodes:

        node.degree_centrality = degree.get(
            node.node_id,
            0.0,
        )

        node.betweenness_centrality = betweenness.get(
            node.node_id,
            0.0,
        )

        node.closeness_centrality = closeness.get(
            node.node_id,
            0.0,
        )

        node.eigenvector_centrality = eigenvector.get(
            node.node_id,
            0.0,
        )

        node.pagerank = pagerank.get(
            node.node_id,
            0.0,
        )

        node.importance_score = compute_importance(
            node
        )

    nodes = graph_state.nodes

    # ------------------------------------------------------
    # Graph Signal Statistics
    # ------------------------------------------------------

    graph_state.signals.centrality.average_degree = (
        sum(n.degree_centrality for n in nodes)
        / len(nodes)
    )

    graph_state.signals.centrality.average_betweenness = (
        sum(n.betweenness_centrality for n in nodes)
        / len(nodes)
    )

    graph_state.signals.centrality.average_closeness = (
        sum(n.closeness_centrality for n in nodes)
        / len(nodes)
    )

    graph_state.signals.centrality.average_eigenvector = (
        sum(n.eigenvector_centrality for n in nodes)
        / len(nodes)
    )

    graph_state.signals.centrality.average_pagerank = (
        sum(n.pagerank for n in nodes)
        / len(nodes)
    )

    # ------------------------------------------------------
    # Importance
    # ------------------------------------------------------

    most_important = max(
        nodes,
        key=lambda n: n.importance_score,
    )

    graph_state.signals.importance.most_important_node = (
        most_important.node_id
    )

    graph_state.signals.importance.highest_importance_score = (
        most_important.importance_score
    )

    graph_state.signals.importance.average_importance = (

        sum(
            n.importance_score
            for n in nodes
        )

        / len(nodes)

    )

    return graph_state