# ==========================================================
# BUILD REASONING READINESS
# ==========================================================

def build_reasoning_ready(
    retrieval_state,
    graph_state,
    evidence_state,
    reasoning_state,
):
    """
    Determines whether the Evidence State is
    ready for Phase-10 reasoning.
    """

    readiness = reasoning_state.readiness

    score = (

        0.45 * evidence_state.evidence_score +

        0.30 * evidence_state.health.health_score +

        0.25 *

        (1.0 - evidence_state.uncertainty.uncertainty_score)

    )

    readiness.score = round(score, 4)

    readiness.ready = (

        readiness.score >= 0.70

    )

    readiness.uncertainty = (

        evidence_state.uncertainty.uncertainty_score

    )

    return reasoning_state