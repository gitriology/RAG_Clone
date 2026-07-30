import numpy as np


# ==========================================================
# COMPUTE EVIDENCE SCORE
# ==========================================================

def compute_evidence_score(
    evidence_state,
):
    """
    Computes the final Evidence State score.
    """

    vector = evidence_state.weighted_vector

    if vector is None or len(vector) == 0:

        evidence_state.evidence_score = 0.0

        return evidence_state

    mean = float(

        np.mean(vector)

    )

    health = evidence_state.signals.health.score

    uncertainty = (

        evidence_state.signals.uncertainty.score

    )

    score = (

        0.60 * mean +

        0.25 * health +

        0.15 * (1.0 - uncertainty)

    )

    evidence_state.evidence_score = round(

        max(0.0, min(score, 1.0)),

        4,

    )

    return evidence_state