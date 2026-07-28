from dataclasses import dataclass, field
from typing import List


# ==========================================================
# SERIALIZED NODE
# ==========================================================

@dataclass
class SerializedNode:

    node_id: str

    text: str

    domain: str

    source: str

    dense_score: float

    sparse_score: float

    rerank_score: float

    hybrid_score: float

    degree: float

    betweenness: float

    closeness: float

    eigenvector: float

    pagerank: float

    importance: float


# ==========================================================
# SERIALIZED EDGE
# ==========================================================

@dataclass
class SerializedEdge:

    source: str

    target: str

    weight: float

    relation: str


# ==========================================================
# SERIALIZED GRAPH STATISTICS
# ==========================================================

@dataclass
class SerializedGraphStatistics:

    total_nodes: int = 0

    total_edges: int = 0

    connected_components: int = 0

    average_degree: float = 0.0

    average_edge_weight: float = 0.0

    graph_density: float = 0.0


# ==========================================================
# SERIALIZED GRAPH SIGNALS
# ==========================================================

@dataclass
class SerializedGraphSignals:

    graph_score: float = 0.0

    coherence_score: float = 0.0

    evidence_quality: float = 0.0

    average_degree: float = 0.0

    average_betweenness: float = 0.0

    average_closeness: float = 0.0

    average_eigenvector: float = 0.0

    average_pagerank: float = 0.0

    graph_consensus: float = 0.0


# ==========================================================
# GRAPH SNAPSHOT
# ==========================================================

@dataclass
class GraphSnapshot:

    version: str = "1.0"

    framework: str = "MS-ARC"

    nodes: List[SerializedNode] = field(default_factory=list)

    edges: List[SerializedEdge] = field(default_factory=list)

    statistics: SerializedGraphStatistics = field(
        default_factory=SerializedGraphStatistics
    )

    signals: SerializedGraphSignals = field(
        default_factory=SerializedGraphSignals
    )