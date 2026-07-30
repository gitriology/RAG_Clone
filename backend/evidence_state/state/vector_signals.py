from dataclasses import dataclass, field


# ==========================================================
# Retrieval
# ==========================================================

@dataclass
class RetrievalVectorSignals:

    agreement: float = 0.0

    margin: float = 0.0

    stability: float = 0.0

    confidence: float = 0.0


# ==========================================================
# Graph
# ==========================================================

@dataclass
class GraphVectorSignals:

    graph_score: float = 0.0

    coherence: float = 0.0

    evidence_quality: float = 0.0

    graph_consensus: float = 0.0


# ==========================================================
# Centrality
# ==========================================================

@dataclass
class CentralityVectorSignals:

    average_degree: float = 0.0

    average_betweenness: float = 0.0

    average_closeness: float = 0.0

    average_eigenvector: float = 0.0

    average_pagerank: float = 0.0


# ==========================================================
# Diagnostics
# ==========================================================

@dataclass
class HealthSignals:

    score: float = 0.0

    completeness: float = 0.0

    consistency: float = 0.0

    balance: float = 0.0


@dataclass
class UncertaintySignals:

    score: float = 0.0

    entropy: float = 0.0

    confidence_gap: float = 0.0


# ==========================================================
# Vector Quality
# ==========================================================

@dataclass
class VectorQualitySignals:

    reliability: float = 0.0


# ==========================================================
# Signal Container
# ==========================================================

@dataclass
class VectorSignals:

    retrieval: RetrievalVectorSignals = field(
        default_factory=RetrievalVectorSignals
    )

    graph: GraphVectorSignals = field(
        default_factory=GraphVectorSignals
    )

    centrality: CentralityVectorSignals = field(
        default_factory=CentralityVectorSignals
    )

    quality: VectorQualitySignals = field(
        default_factory=VectorQualitySignals
    )

    health: HealthSignals = field(
        default_factory=HealthSignals
    )

    uncertainty: UncertaintySignals = field(
        default_factory=UncertaintySignals
    )