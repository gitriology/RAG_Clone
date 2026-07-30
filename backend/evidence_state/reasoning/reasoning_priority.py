# ==========================================================
# BUILD PRIORITY
# ==========================================================

def build_priority(
    evidence_state,
    reasoning_state,
):
    """
    Orders evidence features by importance.
    """

    ordered = sorted(

        evidence_state.features,

        key=lambda feature: feature.importance,

        reverse=True,

    )

    reasoning_state.priority.priority_features = [

        feature.name

        for feature in ordered[:5]

    ]

    return reasoning_state