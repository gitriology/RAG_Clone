"""
Pipeline entry point for MS-ARC.
"""

from backend.ms_arc.state.retrieval_state import RetrievalState

from backend.ms_arc.complexity.analyzer import QueryComplexityAnalyzer

from backend.ms_arc.retrieval.retrieve import retrieve

from backend.ms_arc.agreement.agreement import compute_agreement

from backend.ms_arc.confidence.margin import compute_margin

from backend.ms_arc.stability.stability import compute_stability

from backend.ms_arc.decision.decision_engine import compute_decision


def run_msarc(query: str) -> RetrievalState:

    state = RetrievalState(query=query)

    # ---------------------------------------
    # Phase 7
    # ---------------------------------------

    analyzer = QueryComplexityAnalyzer()

    state = analyzer.analyze(state)

    # ---------------------------------------
    # Retrieval
    # ---------------------------------------

    state = retrieve(state)

    # ---------------------------------------
    # Phase 8
    # ---------------------------------------

    state = compute_agreement(state)

    state = compute_margin(state)

    state = compute_stability(state)

    # ---------------------------------------
    # Phase 9
    # ---------------------------------------

    state = compute_decision(state)

    return state