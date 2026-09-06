from backend.evidence_graph.state.evidence_graph_state import (
    EvidenceGraphState,
)


REQUIRED_IMPORTANCE_METRICS = frozenset(
    {
        "degree",
        "betweenness",
        "closeness",
        "eigenvector",
        "pagerank",
    }
)


def compute_node_importance(
    graph_state: EvidenceGraphState,
):
    """
    Materialize node importance only when explicitly requested.

    Optimization #35 keeps this derived analytics stage lazy. The historical
    importance formula is unchanged and requires all five centrality metrics.
    Callers that do not need graph-aware importance do not pay this cost.
    """
    if graph_state is None:
        raise ValueError("compute_node_importance() received graph_state=None.")

    available = set(graph_state.analytics_metrics_computed)
    missing = REQUIRED_IMPORTANCE_METRICS - available
    if missing:
        raise RuntimeError(
            "Node importance requires graph analytics metrics: "
            f"{sorted(missing)}. Materialize them first via "
            "ensure_graph_analytics(..., metrics=...)."
        )

    if graph_state.nodes and graph_state.signals.importance.average_importance != 0.0:
        return graph_state

    highest = 0.0
    total = 0.0
    best = ""

    for node in graph_state.nodes:
        importance = (
            0.40 * node.rerank_score
            + 0.35 * node.pagerank
            + 0.25 * node.degree_centrality
        )

        node.importance_score = importance
        total += importance

        if importance > highest:
            highest = importance
            best = node.node_id

    average = total / len(graph_state.nodes) if graph_state.nodes else 0.0

    graph_state.signals.importance.highest_importance_score = highest
    graph_state.signals.importance.average_importance = average
    graph_state.signals.importance.most_important_node = best

    return graph_state
