from dataclasses import dataclass, field
from typing import List


# ==========================================================
# AGREEMENT
# ==========================================================

@dataclass
class AgreementMetrics:

    score: float = 0.0

    intersection: int = 0

    union: int = 0


# ==========================================================
# MARGIN
# ==========================================================

@dataclass
class MarginMetrics:

    top_score: float = 0.0

    second_score: float = 0.0

    raw_margin: float = 0.0

    normalized_margin: float = 0.0


# ==========================================================
# STABILITY
# ==========================================================

@dataclass
class StabilityMetrics:

    score: float = 0.0

    overlap_1: float = 0.0

    overlap_2: float = 0.0

    topk_runs: List[int] = field(default_factory=list)



# ==========================================================
# NOVELTY
# ==========================================================

@dataclass
class NoveltyMetrics:

    score: float = 0.0

    novel_documents: int = 0

    duplicate_documents: int = 0


# ==========================================================
# EVIDENCE
# ==========================================================

@dataclass
class EvidenceMetrics:

    score: float = 0.0

    coverage: float = 0.0

    diversity: float = 0.0

    consistency: float = 0.0


# ==========================================================
# DECISION
# ==========================================================


@dataclass
class DecisionMetrics:

    # Final retrieval confidence
    confidence: float = 0.0

    # ACCEPT / EXPAND_TOPK / RERANK_AGAIN / RETRIEVE_AGAIN
    decision: str = ""

    # Human-readable explanation
    reason: str = ""

    # Input signals
    agreement: float = 0.0

    margin: float = 0.0

    stability: float = 0.0

    # Weights used
    agreement_weight: float = 0.40

    margin_weight: float = 0.35

    stability_weight: float = 0.25
# ==========================================================
# SIGNAL CONTAINER
# ==========================================================

@dataclass
class RetrievalSignals:

    agreement: AgreementMetrics = field(
        default_factory=AgreementMetrics
    )

    margin: MarginMetrics = field(
        default_factory=MarginMetrics
    )

    stability: StabilityMetrics = field(
        default_factory=StabilityMetrics
    )

    novelty: NoveltyMetrics = field(
        default_factory=NoveltyMetrics
    )

    evidence: EvidenceMetrics = field(
        default_factory=EvidenceMetrics
    )

    decision: DecisionMetrics = field(
        default_factory=DecisionMetrics
    )
    