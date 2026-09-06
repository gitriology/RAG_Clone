import networkx as nx

from backend.evidence_graph.analytics.centrality import (
    compute_degree,
    compute_betweenness,
    compute_closeness,
    compute_pagerank,
)


# ==========================================================
# Optimization #35 — Metric-level lazy analytics
# ==========================================================

ANALYTIC_METRICS = {
    "degree",
    "betweenness",
    "closeness",
    "eigenvector",
    "pagerank",
}

# Cheap structural metrics are useful for lightweight consumers.
DEFAULT_LAZY_METRICS = frozenset({"degree", "pagerank"})


def compute_eigenvector(graph: nx.Graph):
    """Compute Eigenvector Centrality with a safe convergence fallback."""
    if graph.number_of_nodes() == 0:
        return {}

    try:
        return nx.eigenvector_centrality(
            graph,
            max_iter=1000,
            weight="weight",
        )
    except Exception:
        return {node: 0.0 for node in graph.nodes}


def compute_importance(node):
    """
    Overall node importance.

    This is the historical five-metric importance formula. It is only
    materialized when all metrics required by the formula are available.
    """
    return (
        0.20 * node.degree_centrality
        + 0.20 * node.betweenness_centrality
        + 0.20 * node.closeness_centrality
        + 0.20 * node.eigenvector_centrality
        + 0.20 * node.pagerank
    )


def _requested_metrics(metrics):
    if metrics is None:
        return set(ANALYTIC_METRICS)

    requested = {str(metric).lower() for metric in metrics}
    unknown = requested - ANALYTIC_METRICS
    if unknown:
        raise ValueError(
            f"Unsupported graph analytic metrics: {sorted(unknown)}. "
            f"Expected subset of {sorted(ANALYTIC_METRICS)}."
        )
    return requested


def _store_metric(node, metric, values):
    attribute = {
        "degree": "degree_centrality",
        "betweenness": "betweenness_centrality",
        "closeness": "closeness_centrality",
        "eigenvector": "eigenvector_centrality",
        "pagerank": "pagerank",
    }[metric]
    setattr(node, attribute, values.get(node.node_id, 0.0))


def analyze_graph(graph_state, metrics=None):
    """
    Compute selected graph analytics and update only those signals.

    ``metrics=None`` preserves the historical behavior and computes all five
    metrics. Optimization #35 callers can request only the metrics they need,
    avoiding unnecessary betweenness/closeness/eigenvector computation.

    Supported metrics:
        degree, betweenness, closeness, eigenvector, pagerank
    """
    requested = _requested_metrics(metrics)
    graph = graph_state.graph

    if graph.number_of_nodes() == 0:
        return graph_state

    results = {}

    if "degree" in requested:
        results["degree"] = compute_degree(graph)
    if "betweenness" in requested:
        results["betweenness"] = compute_betweenness(graph)
    if "closeness" in requested:
        results["closeness"] = compute_closeness(graph)
    if "pagerank" in requested:
        results["pagerank"] = compute_pagerank(graph)
    if "eigenvector" in requested:
        results["eigenvector"] = compute_eigenvector(graph)

    for node in graph_state.nodes:
        for metric, values in results.items():
            _store_metric(node, metric, values)

    nodes = graph_state.nodes

    # Signal statistics are updated only for metrics actually computed.
    signal_map = {
        "degree": ("average_degree", "degree_centrality"),
        "betweenness": ("average_betweenness", "betweenness_centrality"),
        "closeness": ("average_closeness", "closeness_centrality"),
        "eigenvector": ("average_eigenvector", "eigenvector_centrality"),
        "pagerank": ("average_pagerank", "pagerank"),
    }

    for metric in requested:
        signal_name, node_attribute = signal_map[metric]
        setattr(
            graph_state.signals.centrality,
            signal_name,
            sum(getattr(n, node_attribute) for n in nodes) / len(nodes),
        )

    # Historical importance requires all five metrics. Do not silently invent
    # partial importance scores when Optimization #35 computes a subset.
    if requested == ANALYTIC_METRICS:
        for node in nodes:
            node.importance_score = compute_importance(node)

        most_important = max(nodes, key=lambda n: n.importance_score)
        graph_state.signals.importance.most_important_node = (
            most_important.node_id
        )
        graph_state.signals.importance.highest_importance_score = (
            most_important.importance_score
        )
        graph_state.signals.importance.average_importance = (
            sum(n.importance_score for n in nodes) / len(nodes)
        )

    return graph_state
