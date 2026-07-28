from dataclasses import dataclass, field

import networkx as nx

from backend.evidence_graph.models.evidence_node import EvidenceNode
from backend.evidence_graph.models.evidence_edge import EvidenceEdge
from backend.evidence_graph.models.evidence_cluster import EvidenceCluster
from backend.evidence_graph.models.graph_statistics import GraphStatistics
from backend.evidence_graph.state.graph_signals import GraphSignals
from backend.evidence_graph.validation.validation_report import (
    GraphValidationReport,
)

@dataclass
class EvidenceGraphState:

    graph: nx.Graph = field(default_factory=nx.Graph)

    nodes: list[EvidenceNode] = field(default_factory=list)

    edges: list[EvidenceEdge] = field(default_factory=list)

    node_lookup: dict[str, EvidenceNode] = field(default_factory=dict)

    edge_lookup: dict[tuple[str, str], EvidenceEdge] = field(default_factory=dict)

    statistics: GraphStatistics = field(
        default_factory=GraphStatistics
    )

    signals: GraphSignals = field(
        default_factory=GraphSignals
    )

    clusters: list[EvidenceCluster] = field(
        default_factory=list
    )

    graph_score: float = 0.0

    ranking = None
    
    validation: GraphValidationReport = field(
        default_factory=GraphValidationReport
    )