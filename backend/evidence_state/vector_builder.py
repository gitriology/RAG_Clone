import numpy as np

from backend.evidence_state.state.evidence_state import (
    EvidenceState,
)


# ==========================================================
# BUILD VECTOR
# ==========================================================

def build_vector(
    evidence_state: EvidenceState,
) -> EvidenceState:
    """
    Builds the normalized Evidence State Vector.

    Uses feature.normalized_value rather than raw values.
    """

    if not evidence_state.features:

        evidence_state.vector = np.array([])

        return evidence_state

    # ------------------------------------------------------
    # Build Vector
    # ------------------------------------------------------

    vector = np.array(

        [

            feature.normalized_value

            for feature in evidence_state.features

        ],

        dtype=float,

    )

    evidence_state.vector = vector

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    stats = evidence_state.statistics

    stats.total_features = len(vector)

    stats.active_features = int(
        np.count_nonzero(vector)
    )

    stats.sparsity = (

        1.0

        -

        stats.active_features

        /

        max(stats.total_features, 1)

    )

    stats.mean_value = float(
        vector.mean()
    )

    stats.max_value = float(
        vector.max()
    )

    stats.min_value = float(
        vector.min()
    )

    stats.variance = float(
        vector.var()
    )

    stats.l2_norm = float(
        np.linalg.norm(vector)
    )

    return evidence_state


# ==========================================================
# WEIGHTED VECTOR STATISTICS
# ==========================================================

def update_weighted_statistics(
    evidence_state: EvidenceState,
) -> EvidenceState:
    """
    Computes statistics for the weighted vector.
    """

    vector = evidence_state.weighted_vector

    if vector is None or len(vector) == 0:

        return evidence_state

    stats = evidence_state.statistics

    stats.weighted_mean = float(
        vector.mean()
    )

    stats.weighted_norm = float(
        np.linalg.norm(vector)
    )

    stats.weighted_variance = float(
        vector.var()
    )

    return evidence_state