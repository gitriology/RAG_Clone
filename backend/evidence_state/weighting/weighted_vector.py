import numpy as np


# ==========================================================
# BUILD WEIGHTED VECTOR
# ==========================================================

def build_weighted_vector(
    evidence_state,
):
    """
    Builds the weighted evidence vector.
    """

    weighted = []

    for feature in evidence_state.features:

        feature.weighted_value = (

            feature.normalized_value *

            feature.weight

        )

        weighted.append(

            feature.weighted_value

        )

    evidence_state.weighted_vector = np.array(

        weighted,

        dtype=float,

    )

    return evidence_state