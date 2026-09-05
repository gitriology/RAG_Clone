"""Tests for Optimization #28: genuine iterative MS-ARC retrieval."""

import pytest

pytest.importorskip("faiss")
import backend.ms_arc.run_msarc as msarc
from backend.ms_arc.state.retrieval_state import RetrievedDocument, RetrievalState


def _fake_analyze(self, state):
    state.query_complexity = 0.9
    state.recommended_topk = 10
    return state


def _fake_retrieve(state, fusion_method="minmax", retrieval_k=None):
    k = int(retrieval_k or state.recommended_topk)
    docs = [
        RetrievedDocument(doc_id=str(i), text=f"doc {i}")
        for i in range(k)
    ]
    state.merged_results = docs
    state.selected_documents = docs
    state.dense_results = docs
    state.sparse_results = docs
    state.debug["retrieval_k_used"] = k
    return state


def _fake_agreement(state):
    # Strong agreement at K>=10, weak at the initial K=5 pass.
    state.signals.agreement.score = 0.90 if state.debug["retrieval_k_used"] >= 10 else 0.80
    return state


def _fake_margin(state):
    state.signals.margin.normalized_margin = 0.90 if state.debug["retrieval_k_used"] >= 10 else 0.80
    state.reranked_results = list(state.selected_documents)
    return state


def _fake_stability(state):
    rankings = state.debug.get("adaptive_rankings", [])
    state.signals.stability.score = 0.50 if len(rankings) == 1 else 0.90
    return state


def _fake_decision(state):
    a = state.signals.agreement.score
    m = state.signals.margin.normalized_margin
    s = state.signals.stability.score
    confidence = 0.40 * a + 0.35 * m + 0.25 * s
    state.retrieval_confidence = confidence
    state.decision = "ACCEPT" if confidence >= 0.75 else "EXPAND_TOPK"
    return state


def test_adaptive_loop_expands_from_5_to_10_then_stops(monkeypatch):
    monkeypatch.setattr(msarc.QueryComplexityAnalyzer, "analyze", _fake_analyze)
    monkeypatch.setattr(msarc, "retrieve", _fake_retrieve)
    monkeypatch.setattr(msarc, "compute_agreement", _fake_agreement)
    monkeypatch.setattr(msarc, "compute_margin", _fake_margin)
    monkeypatch.setattr(msarc, "compute_stability", _fake_stability)
    monkeypatch.setattr(msarc, "compute_decision", _fake_decision)

    state = msarc.run_msarc("complex test query")
    history = state.debug["adaptive_iterations"]

    assert [item["retrieval_k"] for item in history] == [5, 10]
    assert state.debug["adaptive_final_k"] == 10
    assert state.debug["adaptive_iterations_count"] == 2
    assert state.debug["adaptive_stop_reason"] == "confidence_threshold"


def test_adaptive_loop_can_early_exit_at_initial_k(monkeypatch):
    monkeypatch.setattr(msarc.QueryComplexityAnalyzer, "analyze", lambda self, state: state)
    monkeypatch.setattr(msarc, "retrieve", _fake_retrieve)

    def analyze(self, state):
        state.query_complexity = 0.1
        state.recommended_topk = 3
        return state

    def agreement(state):
        state.signals.agreement.score = 1.0
        return state

    def margin(state):
        state.signals.margin.normalized_margin = 1.0
        state.reranked_results = list(state.selected_documents)
        return state

    def stability(state):
        state.signals.stability.score = 1.0
        return state

    monkeypatch.setattr(msarc.QueryComplexityAnalyzer, "analyze", analyze)
    monkeypatch.setattr(msarc, "compute_agreement", agreement)
    monkeypatch.setattr(msarc, "compute_margin", margin)
    monkeypatch.setattr(msarc, "compute_stability", stability)
    monkeypatch.setattr(msarc, "compute_decision", _fake_decision)

    state = msarc.run_msarc("simple query")
    history = state.debug["adaptive_iterations"]

    assert [item["retrieval_k"] for item in history] == [5]
    assert state.debug["adaptive_final_k"] == 5
    assert state.debug["adaptive_stop_reason"] == "confidence_threshold"


def test_adaptive_loop_records_incremental_rankings(monkeypatch):
    monkeypatch.setattr(msarc.QueryComplexityAnalyzer, "analyze", _fake_analyze)
    monkeypatch.setattr(msarc, "retrieve", _fake_retrieve)
    monkeypatch.setattr(msarc, "compute_agreement", _fake_agreement)
    monkeypatch.setattr(msarc, "compute_margin", _fake_margin)
    monkeypatch.setattr(msarc, "compute_stability", _fake_stability)
    monkeypatch.setattr(msarc, "compute_decision", _fake_decision)

    state = msarc.run_msarc("complex test query")

    rankings = state.debug["adaptive_rankings"]
    assert len(rankings) == 2
    assert len(rankings[0]) == 5
    assert len(rankings[1]) == 10
    assert set(rankings[0]).issubset(set(rankings[1]))
