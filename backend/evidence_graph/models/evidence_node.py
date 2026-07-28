from dataclasses import dataclass, field
from typing import List
# ==========================================================
# EVIDENCE NODE
# ==========================================================

@dataclass
class EvidenceNode:
    """
    Represents one retrieved document inside
    the Evidence Graph.
    """

    # Identity
    node_id: str
    text: str
    domain: str
    source: str

    # Retrieval scores
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float = 0.0
    hybrid_score: float = 0.0

    # ======================================================
    # Graph Analytics (Phase 8D)
    # ======================================================

    degree_centrality: float = 0.0

    betweenness_centrality: float = 0.0

    closeness_centrality: float = 0.0

    eigenvector_centrality: float = 0.0

    pagerank: float = 0.0

    importance_score: float = 0.0

    # ======================================================
    # Future Semantic Layer (Phase 8E+)
    # ======================================================

    entities: List[str] = field(default_factory=list)

    claims: List[str] = field(default_factory=list)

    relations: List[str] = field(default_factory=list)

    contradictions: List[str] = field(default_factory=list)