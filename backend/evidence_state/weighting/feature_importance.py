def compute_feature_importance(
    evidence_state,
):
    """
    Importance = normalized value × weight.
    """

    total = 0.0

    for feature in evidence_state.features:

        feature.importance = (

            feature.normalized_value *

            feature.weight

        )

        total += feature.importance

    if total > 0:

        for feature in evidence_state.features:

            feature.importance /= total

    return evidence_state