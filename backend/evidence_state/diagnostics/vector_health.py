from backend.evidence_state.state.evidence_state import (
    VectorHealth,
)


# ==========================================================
# VECTOR HEALTH
# ==========================================================

def compute_vector_health(
    evidence_state,
):
    """
    Computes the health of the evidence vector.
    """

    stats = evidence_state.statistics

    health = VectorHealth()

    # ---------------------------------------------
    # Completeness
    # ---------------------------------------------

    if stats.total_features:

        health.completeness = (

            stats.active_features /

            stats.total_features

        )

    # ---------------------------------------------
    # Consistency
    # ---------------------------------------------

    health.consistency = max(

        0.0,

        1.0 - stats.variance,

    )

    # ---------------------------------------------
    # Balance
    # ---------------------------------------------

    if stats.max_value > 0:

        health.balance = (

            stats.mean_value /

            stats.max_value

        )

    # ---------------------------------------------
    # Final Score
    # ---------------------------------------------

    health.health_score = round(

        (

            health.completeness +

            health.consistency +

            health.balance

        ) / 3,

        4,

    )

    evidence_state.health = health

    evidence_state.signals.health.score = (

        health.health_score

    )

    return evidence_state