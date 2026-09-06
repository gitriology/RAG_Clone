"""Tests for Optimization #37: integrated Evidence State feedback control."""

from pathlib import Path


RUN_MSARC = Path("backend/ms_arc/run_msarc.py").read_text(encoding="utf-8")
RUN_RERANK = Path("backend/reranker/scoring/run_rerank.py").read_text(encoding="utf-8")
RETRIEVAL_STATE = Path("backend/ms_arc/state/retrieval_state.py").read_text(encoding="utf-8")


def test_full_evidence_state_is_part_of_adaptive_feedback_loop():
    assert "build_graph(" in RUN_MSARC
    assert "analytics_mode=\"lazy\"" in RUN_MSARC
    assert "build_evidence_state(" in RUN_MSARC
    assert "controller_evidence_score" in RUN_MSARC
    assert "full_evidence_state_sufficient" in RUN_MSARC


def test_controller_snapshots_are_cleared_and_persisted_on_retrieval_state():
    assert "state.evidence_graph_state = None" in RUN_MSARC
    assert "state.controller_evidence_state = None" in RUN_MSARC
    assert "evidence_graph_state: Any = None" in RETRIEVAL_STATE
    assert "controller_evidence_state: Any = None" in RETRIEVAL_STATE


def test_downstream_pipeline_reuses_controller_graph_and_evidence_state():
    assert "evidence_graph_state" in RUN_RERANK
    assert "controller_evidence_state" in RUN_RERANK
    assert "Reusing Evidence Graph produced by MS-ARC" in RUN_RERANK
    assert "Reusing full Evidence State produced by MS-ARC" in RUN_RERANK
