from backend.evidence_graph.state.evidence_graph_state import (
    EvidenceGraphState,
)


def compute_node_importance(
    graph_state: EvidenceGraphState,
):
    """
    Final importance score.

    Combination of

    • reranker confidence
    • PageRank
    • Degree

    """

    highest = 0.0

    total = 0.0

    best = ""

    for node in graph_state.nodes:

        importance = (

            0.40 * node.rerank_score +

            0.35 * node.pagerank +

            0.25 * node.degree_centrality

        )

        node.importance_score = importance

        total += importance

        if importance > highest:

            highest = importance

            best = node.node_id

    average = 0.0

    if graph_state.nodes:

        average = total / len(graph_state.nodes)

    graph_state.signals.importance.highest_importance = highest

    graph_state.signals.importance.average_importance = average

    graph_state.signals.importance.most_important_node = best

    return graph_state