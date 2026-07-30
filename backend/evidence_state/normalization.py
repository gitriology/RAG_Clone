import numpy as np


# ==========================================================
# NORMALIZE FEATURES
# ==========================================================

def normalize_features(
    evidence_state,
):
    """
    Performs Min-Max normalization on all Evidence Features.

    Raw feature values are preserved.

    Normalized values are stored separately in

        feature.normalized_value
    """

    if not evidence_state.features:
        return evidence_state

    values = np.array(

        [

            feature.value

            for feature in evidence_state.features

        ],

        dtype=float,

    )

    minimum = values.min()

    maximum = values.max()

    # ------------------------------------------------------
    # Constant Vector
    # ------------------------------------------------------

    if maximum == minimum:

        normalized = np.ones_like(values)

    else:

        normalized = (

            values - minimum

        ) / (

            maximum - minimum

        )

    # ------------------------------------------------------
    # Store normalized values
    # ------------------------------------------------------

    for feature, value in zip(

        evidence_state.features,

        normalized,

    ):

        feature.normalized_value = float(value)

    # ------------------------------------------------------
    # Store normalization metadata
    # ------------------------------------------------------

    stats = evidence_state.statistics

    stats.normalization_method = "min-max"

    stats.normalization_min = float(minimum)

    stats.normalization_max = float(maximum)

    stats.normalization_range = float(

        maximum - minimum

    )

    return evidence_state