"""
Pipeline entry point for MS-ARC.
"""

from backend.ms_arc.state.retrieval_state import (
    RetrievalState,
)

from backend.ms_arc.complexity.analyzer import (
    QueryComplexityAnalyzer,
)

from backend.ms_arc.retrieval.retrieve import (
    retrieve,
)

from backend.ms_arc.retrieval.adaptive_policy import (
    AdaptiveRetrievalPolicy,
)

from backend.ms_arc.agreement.agreement import (
    compute_agreement,
)

from backend.ms_arc.confidence.margin import (
    compute_margin,
)

from backend.ms_arc.stability.stability import (
    compute_stability,
)

from backend.ms_arc.decision.decision_engine import (
    compute_decision,
)

from backend.evidence_graph.graph_builder import (
    build_graph,
)

from backend.evidence_state.build_evidence_state import (
    build_evidence_state,
)

from backend.evidence_state.incremental import (
    update_incremental_evidence_state,
    evidence_state_sufficient,
)


def run_msarc(
    query: str,
    fusion_method: str = "minmax",
) -> RetrievalState:
    """
    Execute the complete MS-ARC pipeline.

    fusion_method is explicitly propagated through
    the retrieval pipeline.

    Supported:

        minmax
        rrf
    """

    fusion_method = (
        fusion_method
        or "minmax"
    ).lower().strip()

    if fusion_method not in {
        "minmax",
        "rrf",
    }:

        raise ValueError(
            f"Unsupported fusion method: "
            f"{fusion_method}"
        )

    state = RetrievalState(
        query=query
    )

    # ======================================================
    # Store experiment configuration
    # ======================================================

    state.debug[
        "fusion_method"
    ] = fusion_method

    # ======================================================
    # Phase 7 — Query Complexity
    # ======================================================

    analyzer = (
        QueryComplexityAnalyzer()
    )

    state = analyzer.analyze(
        state
    )

    # ======================================================
    # Optimization #26 / #28 — adaptive retrieval plan
    # ======================================================

    policy = AdaptiveRetrievalPolicy()
    plan = policy.plan(
        query_complexity=state.query_complexity,
        recommended_topk=state.recommended_topk,
    )
    state.debug["adaptive_plan"] = {
        "initial_k": plan.initial_k,
        "expansion_step": plan.expansion_step,
        "max_k": plan.max_k,
        "confidence_threshold": plan.confidence_threshold,
        "convergence_epsilon": plan.convergence_epsilon,
    }

    # ======================================================
    # Optimization #28 — genuine iterative controller
    # ======================================================

    current_k = plan.initial_k
    previous_confidence = None
    iteration = 0
    adaptive_rankings = []
    iteration_history = []

    while True:
        iteration += 1

        state = retrieve(
            state,
            fusion_method=fusion_method,
            retrieval_k=current_k,
        )

        # Any controller graph/state snapshot belongs to the current retrieval
        # depth only. Clear prior-iteration snapshots before evaluating this
        # new evidence set so downstream reuse can never return stale state.
        state.evidence_graph_state = None
        state.controller_evidence_state = None

        current_ids = [
            str(doc.doc_id)
            for doc in state.merged_results
        ]
        adaptive_rankings.append(current_ids)
        state.debug["adaptive_rankings"] = adaptive_rankings

        state = compute_agreement(state)
        state = compute_margin(state)
        state = compute_stability(state)
        state = compute_decision(state)

        # Optimization #36: update only the cheap retrieval-stage EvidenceState
        # after this iteration. The expensive full EvidenceState remains a
        # downstream final-stage operation.
        state.evidence_state = update_incremental_evidence_state(
            state,
            state.evidence_state,
        )
        evidence_score = float(state.evidence_state.evidence_score)
        evidence_sufficient = evidence_state_sufficient(
            state.evidence_state,
            threshold=plan.confidence_threshold,
        )
        state.debug["incremental_evidence_score"] = evidence_score
        state.debug["incremental_evidence_sufficient"] = evidence_sufficient

        confidence = float(state.retrieval_confidence)
        reached_max = current_k >= plan.max_k
        sufficient = policy.should_stop(
            confidence=confidence,
            previous_confidence=previous_confidence,
            current_k=current_k,
            plan=plan,
            evidence_score=evidence_score,
        ) or evidence_sufficient

        # Optimization #37: integrate the complete Evidence State into the
        # adaptive feedback loop only when the cheap retrieval-stage state is
        # still ambiguous. This preserves #36's low-cost early exit while
        # making the full EvidenceState an actual retrieval-control signal.
        full_evidence_score = None
        full_evidence_sufficient = False
        if not sufficient:
            controller_graph_state = build_graph(
                state,
                analytics_mode="lazy",
            )
            controller_evidence_state = build_evidence_state(
                state,
                controller_graph_state,
            )
            state.evidence_graph_state = controller_graph_state
            state.controller_evidence_state = controller_evidence_state
            full_evidence_score = float(
                controller_evidence_state.evidence_score
            )
            full_evidence_sufficient = evidence_state_sufficient(
                controller_evidence_state,
                threshold=plan.confidence_threshold,
            )
            state.debug["controller_evidence_score"] = full_evidence_score
            state.debug["controller_evidence_sufficient"] = full_evidence_sufficient
            state.debug["controller_evidence_state_integrated"] = True
            sufficient = sufficient or full_evidence_sufficient

        if reached_max:
            stop_reason = "max_k_reached"
        elif confidence >= plan.confidence_threshold:
            stop_reason = "confidence_threshold"
        elif full_evidence_sufficient:
            stop_reason = "full_evidence_state_sufficient"
        elif evidence_sufficient:
            stop_reason = "evidence_state_sufficient"
        elif previous_confidence is not None and (
            confidence - previous_confidence
        ) <= plan.convergence_epsilon:
            stop_reason = "confidence_converged"
        else:
            stop_reason = "expand"

        iteration_history.append({
            "iteration": iteration,
            "retrieval_k": current_k,
            "documents": len(state.merged_results),
            "confidence": confidence,
            "evidence_score": evidence_score,
            "evidence_state_sufficient": evidence_sufficient,
            "controller_evidence_score": full_evidence_score,
            "controller_evidence_state_sufficient": full_evidence_sufficient,
            "agreement": float(state.signals.agreement.score),
            "margin": float(state.signals.margin.normalized_margin),
            "stability": float(state.signals.stability.score),
            "decision": state.decision,
            "stop": bool(sufficient),
            "stop_reason": stop_reason,
        })

        print(
            f"[Optimization #28] Iteration {iteration}: "
            f"K={current_k}, confidence={confidence:.4f}, "
            f"decision={state.decision}, action={stop_reason}"
        )

        if sufficient:
            break

        next_k = min(
            current_k + plan.expansion_step,
            plan.max_k,
        )
        if next_k <= current_k:
            break

        previous_confidence = confidence
        current_k = next_k

    state.debug["adaptive_iterations"] = iteration_history
    state.debug["adaptive_final_k"] = current_k
    state.debug["adaptive_iterations_count"] = iteration
    state.debug["adaptive_stop_reason"] = iteration_history[-1]["stop_reason"]
    state.debug["adaptive_documents_considered"] = len(state.merged_results)

    return state