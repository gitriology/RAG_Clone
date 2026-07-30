from dataclasses import dataclass, field
from typing import List


# ==========================================================
# Context
# ==========================================================

@dataclass
class EvidenceContext:

    dominant_domains: List[str] = field(default_factory=list)

    dominant_sources: List[str] = field(default_factory=list)

    cluster_count: int = 0

    largest_cluster_size: int = 0

    graph_density: float = 0.0

    retrieval_confidence: float = 0.0

    graph_score: float = 0.0

    evidence_score: float = 0.0


# ==========================================================
# Summary
# ==========================================================

@dataclass
class EvidenceSummary:

    summary: str = ""


# ==========================================================
# Priority
# ==========================================================

@dataclass
class EvidencePriority:

    priority_features: List[str] = field(default_factory=list)

    secondary_features: List[str] = field(default_factory=list)

    ignored_features: List[str] = field(default_factory=list)


# ==========================================================
# Explanation
# ==========================================================

@dataclass
class EvidenceExplanation:

    explanation: str = ""


# ==========================================================
# Readiness
# ==========================================================

@dataclass
class EvidenceReadiness:

    score: float = 0.0

    uncertainty: float = 0.0


# ==========================================================
# Final Reasoning State
# ==========================================================

@dataclass
class EvidenceReasoningState:

    context: EvidenceContext = field(
        default_factory=EvidenceContext
    )

    summary: EvidenceSummary = field(
        default_factory=EvidenceSummary
    )

    priority: EvidencePriority = field(
        default_factory=EvidencePriority
    )

    explanation: EvidenceExplanation = field(
        default_factory=EvidenceExplanation
    )

    readiness: EvidenceReadiness = field(
        default_factory=EvidenceReadiness
    )