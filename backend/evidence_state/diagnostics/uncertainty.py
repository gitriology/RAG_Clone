import numpy as np

from backend.evidence_state.state.evidence_state import (
    EvidenceUncertainty,
)


# ==========================================================
# UNCERTAINTY
# ==========================================================

def compute_uncertainty(
    evidence_state,
):
    """
    Estimates uncertainty of the evidence vector.
    """

    uncertainty = EvidenceUncertainty()

    vector = evidence_state.weighted_vector

    if vector is None or len(vector) == 0:

        evidence_state.uncertainty = uncertainty

        return evidence_state

    # ---------------------------------------------
    # Entropy
    # ---------------------------------------------

    values = np.abs(vector)

    total = values.sum()

    if total > 0:

        probabilities = values / total

        entropy = -np.sum(

            probabilities *

            np.log2(

                probabilities + 1e-12

            )

        )

    else:

        entropy = 0.0

    uncertainty.entropy = float(entropy)

    # ---------------------------------------------
    # Confidence Gap
    # ---------------------------------------------

    ordered = np.sort(values)

    if len(ordered) >= 2:

        uncertainty.confidence_gap = float(

            ordered[-1] -

            ordered[-2]

        )

    else:

        uncertainty.confidence_gap = 0.0

    # ---------------------------------------------
    # Final Uncertainty
    # ---------------------------------------------

    max_entropy = np.log2(

        max(len(values), 2)

    )

    normalized_entropy = (

        entropy /

        max_entropy

    )

    uncertainty.uncertainty_score = round(

        normalized_entropy,

        4,

    )

    evidence_state.uncertainty = uncertainty

    evidence_state.signals.uncertainty.score = (

        uncertainty.uncertainty_score

    )

    return evidence_state