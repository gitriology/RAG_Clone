from dataclasses import dataclass, field
from typing import List


# ==========================================================
# FEATURE
# ==========================================================

@dataclass
class SerializedFeature:

    name: str

    source: str

    description: str

    value: float

    normalized_value: float

    weight: float

    weighted_value: float

    importance: float


# ==========================================================
# FEATURE GROUP
# ==========================================================

@dataclass
class SerializedFeatureGroup:

    name: str

    score: float

    features: List[str] = field(
        default_factory=list
    )


# ==========================================================
# STATISTICS
# ==========================================================

@dataclass
class SerializedVectorStatistics:

    total_features: int = 0

    active_features: int = 0

    sparsity: float = 0.0

    mean_value: float = 0.0

    max_value: float = 0.0

    min_value: float = 0.0

    variance: float = 0.0

    l2_norm: float = 0.0

    weighted_mean: float = 0.0

    weighted_norm: float = 0.0

    weighted_variance: float = 0.0

    normalization_method: str = ""

    normalization_min: float = 0.0

    normalization_max: float = 0.0


# ==========================================================
# SIGNALS
# ==========================================================

@dataclass
class SerializedSignals:

    evidence_score: float = 0.0

    vector_health: float = 0.0

    uncertainty: float = 0.0


# ==========================================================
# PROFILE
# ==========================================================

@dataclass
class SerializedProfile:

    dominant_features: List[str] = field(default_factory=list)

    weak_features: List[str] = field(default_factory=list)

    uncertainty_features: List[str] = field(default_factory=list)

    profile_score: float = 0.0


# ==========================================================
# REASONING
# ==========================================================

@dataclass
class SerializedReasoning:

    summary: str = ""

    explanation: str = ""

    ready: bool = False

    readiness_score: float = 0.0

    priority_features: List[str] = field(
        default_factory=list
    )


# ==========================================================
# SNAPSHOT
# ==========================================================

@dataclass
class EvidenceSnapshot:

    features: List[SerializedFeature] = field(
        default_factory=list
    )

    groups: List[SerializedFeatureGroup] = field(
        default_factory=list
    )

    vector: List[float] = field(
        default_factory=list
    )

    weighted_vector: List[float] = field(
        default_factory=list
    )

    statistics: SerializedVectorStatistics = field(
        default_factory=SerializedVectorStatistics
    )

    signals: SerializedSignals = field(
        default_factory=SerializedSignals
    )

    profile: SerializedProfile = field(
        default_factory=SerializedProfile
    )

    reasoning: SerializedReasoning = field(
        default_factory=SerializedReasoning
    )