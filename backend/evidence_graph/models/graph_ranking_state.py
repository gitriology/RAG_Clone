from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from backend.evidence_graph.models.evidence_node import (
        EvidenceNode,
    )


@dataclass
class RankedEvidence:

    node: "EvidenceNode"

    graph_score: float = 0.0

    rank: int = 0


@dataclass
class GraphRankingStatistics:

    average_score: float = 0.0

    highest_score: float = 0.0

    lowest_score: float = 0.0


@dataclass
class GraphConsensus:

    score: float = 0.0

    agreement: bool = False

    reranker_top: str = ""

    graph_top: str = ""


@dataclass
class GraphRankingState:

    ranked_nodes: List[RankedEvidence] = field(
        default_factory=list
    )

    statistics: GraphRankingStatistics = field(
        default_factory=GraphRankingStatistics
    )

    consensus: GraphConsensus = field(
        default_factory=GraphConsensus
    )