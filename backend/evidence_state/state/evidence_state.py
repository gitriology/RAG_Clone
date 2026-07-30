from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from backend.evidence_state.state.vector_statistics import (
    VectorStatistics,
)

from backend.evidence_state.state.vector_signals import (
    VectorSignals,
)

from backend.evidence_state.state.reasoning_state import (
    EvidenceReasoningState,
)


# ==========================================================
# FEATURE
# ==========================================================

@dataclass
class EvidenceFeature:

    name: str

    # Raw feature extracted from Phase 7/8
    value: float

    source: str

    description: str = ""

    # Normalized feature
    normalized_value: float = 0.0

    # Weight assigned during weighting stage
    weight: float = 1.0

    weighted_value: float = 0.0

    # Importance after analysis
    importance: float = 0.0


# ==========================================================
# FEATURE GROUP
# ==========================================================
@dataclass
class FeatureGroup:
    """
    Logical collection of related evidence features.
    """

    group_name: str

    features: List[EvidenceFeature] = field(
        default_factory=list
    )

    score: float = 0.0

@dataclass
class EvidenceProfile:
    dominant_features: List[str] = field(default_factory=list)
    weak_features: List[str] = field(default_factory=list)
    uncertainty_features: List[str] = field(default_factory=list)
    profile_score: float = 0.0


@dataclass
class VectorHealth:
    completeness: float = 0.0
    consistency: float = 0.0
    balance: float = 0.0
    health_score: float = 0.0


@dataclass
class EvidenceUncertainty:
    uncertainty_score: float = 0.0
    entropy: float = 0.0
    confidence_gap: float = 0.0

# ==========================================================
# EVIDENCE STATE
# ==========================================================

@dataclass
class EvidenceState:
    """
    Unified Evidence State Vector.

    Phase 9 representation that combines

        Retrieval
        Graph
        Diagnostics

    into one numerical state.
    """

    # ------------------------------------------
    # Raw Vector
    # ------------------------------------------

    vector: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )

    # ------------------------------------------
    # Weighted Vector
    # ------------------------------------------

    weighted_vector: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )

    # ------------------------------------------
    # Typed Features
    # ------------------------------------------

    features: List[EvidenceFeature] = field(
        default_factory=list
    )

    # ------------------------------------------
    # Groups
    # ------------------------------------------

    groups: List[FeatureGroup] = field(
        default_factory=list
    )

    # ------------------------------------------
    # Lookup
    # ------------------------------------------

    feature_lookup: Dict[str, EvidenceFeature] = field(
        default_factory=dict
    )
    # ------------------------------------------
    # Group Lookup
    # ------------------------------------------

    group_lookup: Dict[str, FeatureGroup] = field(
        default_factory=dict
    )
    # ------------------------------------------
    # Statistics
    # ------------------------------------------

    statistics: VectorStatistics = field(
        default_factory=VectorStatistics
    )

    # ------------------------------------------
    # Signals
    # ------------------------------------------

    signals: VectorSignals = field(
        default_factory=VectorSignals
    )

    # ------------------------------------------
    # Overall Evidence Score
    # ------------------------------------------

    evidence_score: float = 0.0

    # ------------------------------------------
    # Reasoning
    # ------------------------------------------

    reasoning: EvidenceReasoningState = field(
        default_factory=EvidenceReasoningState
    )
    profile: EvidenceProfile = field(
        default_factory=EvidenceProfile
    )

    health: VectorHealth = field(
        default_factory=VectorHealth
    )

    uncertainty: EvidenceUncertainty = field(
        default_factory=EvidenceUncertainty
    )

    def get_group(
        self,
        name: str,
    ):
        """
        Returns a feature group by name.
        """

        return self.group_lookup.get(name)