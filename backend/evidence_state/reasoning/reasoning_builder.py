from backend.evidence_state.state.reasoning_state import (
    EvidenceReasoningState,
)

from backend.evidence_state.reasoning.build_context import (
    build_context,
)

from backend.evidence_state.reasoning.reasoning_summary import (
    build_summary,
)

from backend.evidence_state.reasoning.reasoning_priority import (
    build_priority,
)

from backend.evidence_state.reasoning.reasoning_explanation import (
    build_explanation,
)

from backend.evidence_state.reasoning.reasoning_ready import (
    build_reasoning_ready,
)


# ==========================================================
# BUILD REASONING STATE
# ==========================================================

def build_reasoning_state(
    retrieval_state,
    graph_state,
    evidence_state,
):
    """
    Builds the complete reasoning state used as the
    interface to the Phase-10 Reasoning Engine.
    """

    reasoning_state = EvidenceReasoningState()

    reasoning_state = build_context(
        retrieval_state,
        graph_state,
        evidence_state,
        reasoning_state,
    )

    reasoning_state = build_summary(
        retrieval_state,
        graph_state,
        evidence_state,
        reasoning_state,
    )

    reasoning_state = build_priority(
        evidence_state,
        reasoning_state,
    )

    reasoning_state = build_explanation(
        retrieval_state,
        graph_state,
        evidence_state,
        reasoning_state,
    )

    reasoning_state = build_reasoning_ready(
        retrieval_state,
        graph_state,
        evidence_state,
        reasoning_state,
    )

    evidence_state.reasoning = reasoning_state

    return evidence_state