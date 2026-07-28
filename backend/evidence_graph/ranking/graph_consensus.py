from typing import List

from backend.evidence_graph.models.evidence_node import (
    EvidenceNode,
)

from backend.evidence_graph.models.graph_ranking_state import (
    GraphConsensus,
    RankedEvidence,
)


def compute_consensus(
    reranked_nodes: List[EvidenceNode],
    ranked_nodes: List[RankedEvidence],
) -> GraphConsensus:
    """
    Computes the agreement between:

        • CrossEncoder ranking
        • Graph-aware ranking

    Since graph_state.nodes are inserted from
    retrieval_state.reranked_results, the first
    EvidenceNode represents the CrossEncoder's
    top-ranked document.
    """

    consensus = GraphConsensus()

    if not reranked_nodes or not ranked_nodes:
        return consensus

    # --------------------------------------------------
    # Top node according to CrossEncoder
    # --------------------------------------------------

    reranker_top = reranked_nodes[0].node_id

    # --------------------------------------------------
    # Top node according to Graph Ranking
    # --------------------------------------------------

    graph_top = ranked_nodes[0].node.node_id

    # --------------------------------------------------
    # Populate metrics
    # --------------------------------------------------

    consensus.reranker_top = reranker_top
    consensus.graph_top = graph_top

    consensus.agreement = (
        reranker_top == graph_top
    )

    consensus.score = (
        1.0 if consensus.agreement else 0.0
    )

    return consensus