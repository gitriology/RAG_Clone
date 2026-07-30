import json

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
# LOAD SNAPSHOT
# ==========================================================

def load_snapshot(
    filename: str,
) -> EvidenceSnapshot:
    """
    Loads an EvidenceSnapshot from JSON.

    This reconstructs only the serialized snapshot.
    Runtime objects are rebuilt separately by restore.py.
    """

    with open(

        filename,

        "r",

        encoding="utf-8",

    ) as file:

        data = json.load(file)

    snapshot = EvidenceSnapshot()

    # ======================================================
    # Features
    # ======================================================

    for feature in data.get(

        "features",

        [],

    ):

        snapshot.features.append(

            SerializedFeature(

                name=feature["name"],

                source=feature["source"],

                description=feature["description"],

                value=feature["value"],

                normalized_value=feature["normalized_value"],

                weight=feature["weight"],

                weighted_value=feature["weighted_value"],

                importance=feature["importance"],

            )

        )

    # ======================================================
    # Feature Groups
    # ======================================================

    for group in data.get(

        "groups",

        [],

    ):

        snapshot.groups.append(

            SerializedFeatureGroup(

                name=group["name"],

                score=group["score"],

                features=group.get(

                    "features",

                    [],

                ),

            )

        )

    # ======================================================
    # Vector
    # ======================================================

    snapshot.vector = data.get(

        "vector",

        [],

    )

    # ======================================================
    # Weighted Vector
    # ======================================================

    snapshot.weighted_vector = data.get(

        "weighted_vector",

        [],

    )

    # ======================================================
    # Statistics
    # ======================================================

    stats = data.get(

        "statistics",

        {},

    )

    snapshot.statistics = SerializedVectorStatistics(

        total_features=stats.get(

            "total_features",

            0,

        ),

        active_features=stats.get(

            "active_features",

            0,

        ),

        sparsity=stats.get(

            "sparsity",

            0.0,

        ),

        mean_value=stats.get(

            "mean_value",

            0.0,

        ),

        max_value=stats.get(

            "max_value",

            0.0,

        ),

        min_value=stats.get(

            "min_value",

            0.0,

        ),

        variance=stats.get(

            "variance",

            0.0,

        ),

        l2_norm=stats.get(

            "l2_norm",

            0.0,

        ),

        weighted_mean=stats.get(

            "weighted_mean",

            0.0,

        ),

        weighted_norm=stats.get(

            "weighted_norm",

            0.0,

        ),

        weighted_variance=stats.get(

            "weighted_variance",

            0.0,

        ),

        normalization_method=stats.get(

            "normalization_method",

            "",

        ),

        normalization_min=stats.get(

            "normalization_min",

            0.0,

        ),

        normalization_max=stats.get(

            "normalization_max",

            0.0,

        ),

    )

    # ======================================================
    # Signals
    # ======================================================

    signals = data.get(

        "signals",

        {},

    )

    snapshot.signals = SerializedSignals(

        evidence_score=signals.get(

            "evidence_score",

            0.0,

        ),

        vector_health=signals.get(

            "vector_health",

            0.0,

        ),

        uncertainty=signals.get(

            "uncertainty",

            0.0,

        ),

    )

    # ======================================================
    # Profile
    # ======================================================

    profile = data.get(

        "profile",

        {},

    )

    snapshot.profile = SerializedProfile(

        dominant_features=profile.get(

            "dominant_features",

            [],

        ),

        weak_features=profile.get(

            "weak_features",

            [],

        ),

        uncertainty_features=profile.get(

            "uncertainty_features",

            [],

        ),

        profile_score=profile.get(

            "profile_score",

            0.0,

        ),

    )

    # ======================================================
    # Reasoning
    # ======================================================

    reasoning = data.get(

        "reasoning",

        {},

    )

    snapshot.reasoning = SerializedReasoning(

        summary=reasoning.get(

            "summary",

            "",

        ),

        explanation=reasoning.get(

            "explanation",

            "",

        ),

        ready=reasoning.get(

            "ready",

            False,

        ),

        readiness_score=reasoning.get(

            "readiness_score",

            0.0,

        ),

        priority_features=reasoning.get(

            "priority_features",

            [],

        ),

    )

    return snapshot