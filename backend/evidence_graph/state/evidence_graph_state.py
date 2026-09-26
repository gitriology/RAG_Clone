from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from typing import List, Dict, Tuple, Set

import networkx as nx

from backend.evidence_graph.models.evidence_node import (
    EvidenceNode,
)

from backend.evidence_graph.models.evidence_edge import (
    EvidenceEdge,
)

from backend.evidence_graph.models.evidence_cluster import (
    EvidenceCluster,
)

from backend.evidence_graph.models.graph_statistics import (
    GraphStatistics,
)

from backend.evidence_graph.state.graph_signals import (
    GraphSignals,
)

from backend.evidence_graph.validation.validation_report import (
    GraphValidationReport,
)


@dataclass
class EvidenceGraphState:

    # ======================================================
    # NETWORKX GRAPH
    # ======================================================

    graph: nx.Graph = field(
        default_factory=nx.Graph
    )

    # ======================================================
    # GRAPH NODES
    # ======================================================

    nodes: List[EvidenceNode] = field(
        default_factory=list
    )

    # ======================================================
    # GRAPH EDGES
    # ======================================================

    edges: List[EvidenceEdge] = field(
        default_factory=list
    )

    # ======================================================
    # LOOKUPS
    # ======================================================

    node_lookup: Dict[str, EvidenceNode] = field(
        default_factory=dict
    )

    edge_lookup: Dict[
        Tuple[str, str],
        EvidenceEdge,
    ] = field(
        default_factory=dict
    )

    # ======================================================
    # STATISTICS
    # ======================================================

    statistics: GraphStatistics = field(
        default_factory=GraphStatistics
    )

    # ======================================================
    # SIGNALS
    # ======================================================

    signals: GraphSignals = field(
        default_factory=GraphSignals
    )

    # ======================================================
    # CLUSTERS
    # ======================================================

    clusters: List[EvidenceCluster] = field(
        default_factory=list
    )

    # ======================================================
    # GRAPH SCORE
    # ======================================================

    graph_score: float = 0.0

    # ======================================================
    # GRAPH RANKING
    # ======================================================

    ranking: Optional[object] = None

    # ======================================================
    # OPTIMIZATION #34 — LAZY ANALYTICS STATE
    # ======================================================

    analytics_mode: str = "full"

    analytics_computed: bool = False

    analytics_metrics_computed: Set[str] = field(default_factory=set)

    analytics_sufficiency: bool = False

    analytics_skip_reason: str = ""

    # ======================================================
    # VALIDATION
    # ======================================================

    validation: GraphValidationReport = field(
        default_factory=GraphValidationReport
    )