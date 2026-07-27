from backend.ms_arc.state.retrieval_state import RetrievalState
from backend.ms_arc.state.retrieval_signals import DecisionMetrics


# ==========================================================
# DECISION ENGINE
# ==========================================================

def compute_decision(
    state: RetrievalState,
) -> RetrievalState:
    """
    Computes the final retrieval decision from
    Agreement, Margin and Stability.

    Produces

    state.signals.decision
    """

    # =====================================================
    # Read Signals
    # =====================================================

    agreement = state.signals.agreement.score

    margin = state.signals.margin.normalized_margin

    stability = state.signals.stability.score

    # =====================================================
    # Static Weights
    # =====================================================

    agreement_weight = 0.40

    margin_weight = 0.35

    stability_weight = 0.25

    # =====================================================
    # Confidence
    # =====================================================

    confidence = (

        agreement * agreement_weight +

        margin * margin_weight +

        stability * stability_weight

    )

    confidence = max(
        0.0,
        min(1.0, confidence)
    )

    # =====================================================
    # Decision
    # =====================================================

    if confidence >= 0.75:

        decision = "ACCEPT"

    elif confidence >= 0.55:

        decision = "EXPAND_TOPK"

    elif confidence >= 0.35:

        decision = "RERANK_AGAIN"

    else:

        decision = "RETRIEVE_AGAIN"

    # =====================================================
    # Reason Builder
    # =====================================================

    reasons = []

    if agreement < 0.25:
        reasons.append("Low agreement")

    if margin < 0.20:
        reasons.append("Ambiguous reranking")

    if stability < 0.60:
        reasons.append("Unstable retrieval")

    if not reasons:
        reasons.append("High retrieval confidence")

    reason = ", ".join(reasons)

    # =====================================================
    # Store Typed Metrics
    # =====================================================

    state.signals.decision = DecisionMetrics(

        confidence=confidence,

        decision=decision,

        reason=reason,

        agreement=agreement,

        margin=margin,

        stability=stability,

        agreement_weight=agreement_weight,

        margin_weight=margin_weight,

        stability_weight=stability_weight,

    )

    return state