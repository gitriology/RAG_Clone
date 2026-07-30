import json
from dataclasses import asdict

import numpy as np

from backend.evidence_state.serialization.evidence_snapshot import (
    EvidenceSnapshot,
    SerializedFeature,
    SerializedFeatureGroup,
    SerializedVectorStatistics,
    SerializedSignals,
    SerializedProfile,
    SerializedReasoning,
)


# ==========================================================
# NUMPY → PYTHON CONVERTER
# ==========================================================

def to_python(obj):
    """
    Converts NumPy scalar types into native Python
    types so that json.dump() can serialize them.
    """

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    raise TypeError(
        f"Object of type {type(obj).__name__} "
        "is not JSON serializable"
    )


# ==========================================================
# BUILD SNAPSHOT
# ==========================================================

def build_snapshot(
    evidence_state,
) -> EvidenceSnapshot:
    """
    Converts the complete EvidenceState into a
    serializable snapshot.
    """

    snapshot = EvidenceSnapshot()

    # ======================================================
    # Features
    # ======================================================

    for feature in evidence_state.features:

        snapshot.features.append(

            SerializedFeature(

                name=feature.name,

                source=feature.source,

                description=feature.description,

                value=feature.value,

                normalized_value=feature.normalized_value,

                weight=feature.weight,

                weighted_value=feature.weighted_value,

                importance=feature.importance,

            )

        )

    # ======================================================
    # Feature Groups
    # ======================================================

    for group in evidence_state.groups:

        snapshot.groups.append(

            SerializedFeatureGroup(

                name=group.group_name,

                score=sum(

                    feature.importance

                    for feature in group.features

                ),

                features=[

                    feature.name

                    for feature in group.features

                ],

            )

        )

    # ======================================================
    # Vector
    # ======================================================

    if evidence_state.vector is not None:

        snapshot.vector = (

            evidence_state.vector.tolist()

        )

    # ======================================================
    # Weighted Vector
    # ======================================================

    if evidence_state.weighted_vector is not None:

        snapshot.weighted_vector = (

            evidence_state.weighted_vector.tolist()

        )

    # ======================================================
    # Statistics
    # ======================================================

    stats = evidence_state.statistics

    snapshot.statistics = SerializedVectorStatistics(

        total_features=stats.total_features,

        active_features=stats.active_features,

        sparsity=stats.sparsity,

        mean_value=stats.mean_value,

        max_value=stats.max_value,

        min_value=stats.min_value,

        variance=stats.variance,

        l2_norm=stats.l2_norm,

        weighted_mean=stats.weighted_mean,

        weighted_norm=stats.weighted_norm,

        weighted_variance=stats.weighted_variance,

        normalization_method=stats.normalization_method,

        normalization_min=stats.normalization_min,

        normalization_max=stats.normalization_max,

    )

    # ======================================================
    # Signals
    # ======================================================

    snapshot.signals = SerializedSignals(

        evidence_score=evidence_state.evidence_score,

        vector_health=evidence_state.health.health_score,

        uncertainty=evidence_state.uncertainty.uncertainty_score,

    )

    # ======================================================
    # Profile
    # ======================================================

    profile = evidence_state.profile

    snapshot.profile = SerializedProfile(

        dominant_features=profile.dominant_features,

        weak_features=profile.weak_features,

        uncertainty_features=profile.uncertainty_features,

        profile_score=profile.profile_score,

    )

    # ======================================================
    # Reasoning
    # ======================================================

    reasoning = evidence_state.reasoning

    snapshot.reasoning = SerializedReasoning(

        summary=reasoning.summary.summary,

        explanation=reasoning.explanation.explanation,

        ready=reasoning.readiness.ready,

        readiness_score=reasoning.readiness.score,

        priority_features=reasoning.priority.priority_features,

    )

    return snapshot


# ==========================================================
# SAVE SNAPSHOT
# ==========================================================

def save_evidence_state(
    evidence_state,
    filename: str,
):
    """
    Saves the complete EvidenceState as JSON.
    """

    snapshot = build_snapshot(
        evidence_state
    )

    with open(

        filename,

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            asdict(snapshot),

            file,

            indent=4,

            default=to_python,

        )