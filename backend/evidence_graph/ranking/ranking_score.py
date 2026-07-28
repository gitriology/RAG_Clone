from backend.evidence_graph.models.evidence_node import (
    EvidenceNode,
)


def compute_node_graph_score(
    node: EvidenceNode,
    coherence: float,
):
    """
    Graph-aware ranking score.

    All components are normalized to [0,1].
    """

    rerank = max(
        0.0,
        min(1.0, (node.rerank_score + 15.0) / 15.0)
    )

    score = (

        0.45 * rerank +

        0.25 * node.importance_score +

        0.20 * node.pagerank +

        0.10 * coherence

    )

    return score