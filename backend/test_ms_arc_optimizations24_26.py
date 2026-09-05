"""Regression tests for MS-ARC Optimizations #24, #25 and #26."""

from backend.ms_arc.complexity.analyzer import QueryComplexityAnalyzer
from backend.ms_arc.retrieval.adaptive_policy import AdaptiveRetrievalPolicy
from backend.ms_arc.state.retrieval_state import RetrievalState


def test_complexity_lightweight_preserves_four_features_without_spacy():
    state = RetrievalState(query="How does FAISS retrieval compare with BM25 ranking?")
    state = QueryComplexityAnalyzer(ner_mode="lightweight").analyze(state)

    details = state.debug["complexity"]
    assert details["entity_signal_mode"] == "lightweight"
    assert details["query_length"] > 0
    assert "technical_score" in details
    assert "multihop_score" in details
    assert 0.0 <= state.query_complexity <= 1.0
    assert state.recommended_topk >= 8


def test_complexity_keeps_research_signal_and_is_not_removed():
    simple = QueryComplexityAnalyzer(ner_mode="lightweight").analyze(
        RetrievalState(query="What is FAISS?")
    )
    complex_state = QueryComplexityAnalyzer(ner_mode="lightweight").analyze(
        RetrievalState(query="Compare FAISS and BM25 retrieval latency, precision, recall, and ranking stability")
    )

    assert simple.query_complexity >= 0.0
    assert complex_state.query_complexity > simple.query_complexity
    assert complex_state.query_type in {"medium", "complex", "very_complex"}


def test_adaptive_policy_starts_small_and_controls_expansion():
    policy = AdaptiveRetrievalPolicy()
    simple = policy.plan(0.10, 3)
    complex_plan = policy.plan(0.90, 10)

    assert simple.initial_k == 5
    assert complex_plan.initial_k == 5
    assert complex_plan.max_k > simple.max_k
    assert complex_plan.expansion_step == 5
    assert policy.should_stop(0.80, current_k=5, plan=complex_plan)
    assert policy.should_stop(0.50, previous_confidence=0.49, current_k=10, plan=complex_plan)
