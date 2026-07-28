from typing import List

from backend.evidence_graph.models.evidence_node import (
    EvidenceNode,
)
from backend.evidence_graph.models.evidence_edge import (
    EvidenceEdge,
)


# ==========================================================
# SIMILARITY
# ==========================================================

def compute_similarity(
    node1: EvidenceNode,
    node2: EvidenceNode,
) -> float:
    """
    Lightweight similarity between two evidence nodes.

    Current signals:
        • same domain
        • reranker confidence similarity

    Future versions:
        • BGE cosine similarity
        • claim overlap
        • entity overlap
        • contradiction penalty
    """

    score = 0.0

    # ------------------------------------------------------
    # Same domain
    # ------------------------------------------------------

    if node1.domain == node2.domain:
        score += 0.50

    # ------------------------------------------------------
    # Similar reranker confidence
    # ------------------------------------------------------

    difference = abs(
        node1.rerank_score -
        node2.rerank_score
    )

    score += max(
        0.0,
        0.50 - difference
    )

    return min(score, 1.0)


# ==========================================================
# BUILD EDGES
# ==========================================================

def build_edges(
    nodes: List[EvidenceNode],
    similarity_threshold: float = 0.55,
) -> List[EvidenceEdge]:
    """
    Creates weighted semantic edges between nodes.
    """

    edges: List[EvidenceEdge] = []

    for i in range(len(nodes)):

        for j in range(i + 1, len(nodes)):

            similarity = compute_similarity(
                nodes[i],
                nodes[j],
            )

            if similarity >= similarity_threshold:

                edges.append(

                    EvidenceEdge(

                        source=nodes[i].node_id,

                        target=nodes[j].node_id,

                        weight=float(similarity),

                        relation="semantic_similarity",

                    )

                )

    return edges