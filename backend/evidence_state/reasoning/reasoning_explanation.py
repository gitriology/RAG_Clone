# ==========================================================
# BUILD EXPLANATION
# ==========================================================

def build_explanation(
    retrieval_state,
    graph_state,
    evidence_state,
    reasoning_state,
):
    """
    Generates a concise explanation of why the
    evidence received its score.
    """

    explanation = reasoning_state.explanation

    profile = evidence_state.profile

    dominant = ", ".join(
        profile.dominant_features
    )

    explanation.explanation = (

        "Evidence quality is primarily influenced by "

        f"{dominant}. "

        f"Overall evidence score = "

        f"{evidence_state.evidence_score:.3f}."

    )

    return reasoning_state