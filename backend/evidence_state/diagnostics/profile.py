from backend.evidence_state.state.evidence_state import (
    EvidenceProfile,
)


# ==========================================================
# BUILD EVIDENCE PROFILE
# ==========================================================

def build_profile(
    evidence_state,
):
    """
    Builds an interpretable profile of the evidence.
    """

    profile = EvidenceProfile()

    if not evidence_state.features:

        evidence_state.profile = profile

        return evidence_state

    ordered = sorted(

        evidence_state.features,

        key=lambda f: f.importance,

        reverse=True,

    )

    profile.dominant_features = [

        feature.name

        for feature in ordered[:5]

    ]

    profile.weak_features = [

        feature.name

        for feature in ordered[-3:]

    ]

    profile.profile_score = round(

        evidence_state.evidence_score,

        4,

    )

    evidence_state.profile = profile

    return evidence_state