import numpy as np

from backend.evidence_state.state.evidence_state import (
    EvidenceState,
    EvidenceFeature,
    FeatureGroup,
)

from backend.evidence_state.serialization.evidence_snapshot import (
    EvidenceSnapshot,
)


# ==========================================================
# RESTORE EVIDENCE STATE
# ==========================================================

def restore_evidence_state(
    snapshot: EvidenceSnapshot,
) -> EvidenceState:
    """
    Restores a complete runtime EvidenceState
    from a serialized EvidenceSnapshot.
    """

    state = EvidenceState()

    # ======================================================
    # Features
    # ======================================================

    for feature in snapshot.features:

        restored = EvidenceFeature(

            name=feature.name,

            value=feature.value,

            source=feature.source,

            description=feature.description,

            normalized_value=feature.normalized_value,

            weight=feature.weight,

            weighted_value=feature.weighted_value,

            importance=feature.importance,

        )

        state.features.append(

            restored

        )

        state.feature_lookup[

            restored.name

        ] = restored

    # ======================================================
    # Feature Groups
    # ======================================================

    for group in snapshot.groups:

        restored_group = FeatureGroup(

            group_name=group.name,

        )

        for feature_name in group.features:

            if feature_name in state.feature_lookup:

                restored_group.features.append(

                    state.feature_lookup[

                        feature_name

                    ]

                )

        state.groups.append(

            restored_group

        )
        state.group_lookup[
            restored_group.group_name
        ] = restored_group

    # ======================================================
    # Vector
    # ======================================================

    state.vector = np.array(

        snapshot.vector,

        dtype=float,

    )

    # ======================================================
    # Weighted Vector
    # ======================================================

    state.weighted_vector = np.array(

        snapshot.weighted_vector,

        dtype=float,

    )

    # ======================================================
    # Statistics
    # ======================================================

    stats = snapshot.statistics

    state.statistics.total_features = (

        stats.total_features

    )

    state.statistics.active_features = (

        stats.active_features

    )

    state.statistics.sparsity = (

        stats.sparsity

    )

    state.statistics.mean_value = (

        stats.mean_value

    )

    state.statistics.max_value = (

        stats.max_value

    )

    state.statistics.min_value = (

        stats.min_value

    )

    state.statistics.variance = (

        stats.variance

    )

    state.statistics.l2_norm = (

        stats.l2_norm

    )

    state.statistics.weighted_mean = (

        stats.weighted_mean

    )

    state.statistics.weighted_norm = (

        stats.weighted_norm

    )

    state.statistics.weighted_variance = (

        stats.weighted_variance

    )

    state.statistics.normalization_method = (

        stats.normalization_method

    )

    state.statistics.normalization_min = (

        stats.normalization_min

    )

    state.statistics.normalization_max = (

        stats.normalization_max

    )

    # ======================================================
    # Signals
    # ======================================================

    state.evidence_score = (

        snapshot.signals.evidence_score

    )

    state.health.health_score = (

        snapshot.signals.vector_health

    )

    state.signals.health.score = (

        snapshot.signals.vector_health

    )

    state.uncertainty.uncertainty_score = (

        snapshot.signals.uncertainty

    )

    state.signals.uncertainty.score = (

        snapshot.signals.uncertainty

    )

    # ======================================================
    # Profile
    # ======================================================

    profile = snapshot.profile

    state.profile.dominant_features = (

        profile.dominant_features

    )

    state.profile.weak_features = (

        profile.weak_features

    )

    state.profile.uncertainty_features = (

        profile.uncertainty_features

    )

    state.profile.profile_score = (

        profile.profile_score

    )

    # ======================================================
    # Reasoning
    # ======================================================

    reasoning = snapshot.reasoning

    state.reasoning.summary.summary = (

        reasoning.summary

    )

    state.reasoning.explanation.explanation = (

        reasoning.explanation

    )

    state.reasoning.priority.priority_features = (

        reasoning.priority_features

    )

    state.reasoning.readiness.ready = (

        reasoning.ready

    )

    state.reasoning.readiness.score = (

        reasoning.readiness_score

    )

    return state