from dataclasses import dataclass, field
from typing import List


# ==========================================================
# GRAPH METRICS
# ==========================================================

@dataclass
class GraphMetrics:

    node_count: int = 0

    edge_count: int = 0

    graph_density: float = 0.0

    average_degree: float = 0.0

    connected_components: int = 0


# ==========================================================
# CLAIM METRICS
# ==========================================================

@dataclass
class ClaimMetrics:

    total_claims: int = 0

    average_claim_length: float = 0.0


# ==========================================================
# ENTITY METRICS
# ==========================================================

@dataclass
class EntityMetrics:

    total_entities: int = 0

    unique_entities: int = 0


# ==========================================================
# RELATION METRICS
# ==========================================================

@dataclass
class RelationMetrics:

    support_links: int = 0

    contradiction_links: int = 0

    reference_links: int = 0

    same_entity_links: int = 0


# ==========================================================
# PROVENANCE METRICS
# ==========================================================

@dataclass
class ProvenanceMetrics:

    source_count: int = 0

    linked_claims: int = 0


# ==========================================================
# COHERENCE
# ==========================================================

@dataclass
class CoherenceMetrics:

    score: float = 0.0

    connected_component_ratio: float = 0.0

    average_cluster_score: float = 0.0


# ==========================================================
# EVIDENCE QUALITY
# ==========================================================

@dataclass
class EvidenceQualityMetrics:

    score: float = 0.0

    retrieval_confidence: float = 0.0

    graph_score: float = 0.0

    coherence_score: float = 0.0


# ==========================================================
# CENTRALITY METRICS
# ==========================================================

@dataclass
class CentralityMetrics:
    """
    Aggregate graph-centrality statistics.
    """

    average_degree: float = 0.0

    average_betweenness: float = 0.0

    average_closeness: float = 0.0

    average_eigenvector: float = 0.0

    average_pagerank: float = 0.0


# ==========================================================
# IMPORTANCE METRICS
# ==========================================================

@dataclass
class ImportanceMetrics:
    """
    Global importance statistics.
    """

    most_important_node: str = ""

    highest_importance_score: float = 0.0

    average_importance: float = 0.0
@dataclass
class RankingMetrics:

    average_score: float = 0.0

    highest_score: float = 0.0

    lowest_score: float = 0.0

    graph_consensus: float = 0.0

# ==========================================================
# GRAPH SIGNAL CONTAINER
# ==========================================================

@dataclass
class GraphSignals:

    graph: GraphMetrics = field(
        default_factory=GraphMetrics
    )

    claims: ClaimMetrics = field(
        default_factory=ClaimMetrics
    )

    entities: EntityMetrics = field(
        default_factory=EntityMetrics
    )

    relations: RelationMetrics = field(
        default_factory=RelationMetrics
    )

    provenance: ProvenanceMetrics = field(
        default_factory=ProvenanceMetrics
    )

    coherence: CoherenceMetrics = field(
        default_factory=CoherenceMetrics
    )

    quality: EvidenceQualityMetrics = field(
        default_factory=EvidenceQualityMetrics
    )

    centrality: CentralityMetrics = field(
        default_factory=CentralityMetrics
    )

    importance: ImportanceMetrics = field(
        default_factory=ImportanceMetrics
    )
    ranking: RankingMetrics = field(
        default_factory=RankingMetrics
    )