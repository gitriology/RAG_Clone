# ==========================================================
# FEATURE DIAGNOSTICS
# ==========================================================

def diagnose_features(
    evidence_state,
):
    """
    Identifies uncertain features based on importance.
    """

    profile = evidence_state.profile

    profile.uncertainty_features = [

        feature.name

        for feature in evidence_state.features

        if feature.importance < 0.05

    ]

    return evidence_state