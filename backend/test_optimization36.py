import numpy as np

from backend.evidence_state.incremental import (
    evidence_state_sufficient,
    update_incremental_evidence_state,
)
from backend.evidence_state.state.evidence_state import EvidenceState
from backend.ms_arc.state.retrieval_state import RetrievalState


def _state():
    state = RetrievalState(query="What is satellite communication?")
    state.query_complexity = 0.2
    state.recommended_topk = 5
    state.merged_results = []
    state.signals.agreement.score = 0.95
    state.signals.margin.normalized_margin = 0.95
    state.signals.stability.score = 0.95
    state.signals.novelty.score = 0.8
    state.signals.evidence.score = 0.9
    state.signals.evidence.coverage = 0.9
    state.signals.evidence.diversity = 0.8
    state.signals.evidence.consistency = 0.9
    state.retrieval_confidence = 0.95
    return state


def test_incremental_update_only_builds_retrieval_features():
    retrieval = _state()
    evidence = update_incremental_evidence_state(retrieval)

    assert isinstance(evidence, EvidenceState)
    assert evidence.debug["update_mode"] == "incremental_retrieval"
    assert evidence.debug["feature_count"] == 11
    assert all(feature.source == "retrieval" for feature in evidence.features)
    assert evidence.vector.shape == (11,)
    assert evidence.weighted_vector.shape == (11,)


def test_incremental_state_is_reused_and_refreshed():
    retrieval = _state()
    evidence = update_incremental_evidence_state(retrieval)
    first_vector = evidence.vector.copy()

    retrieval.retrieval_confidence = 0.4
    retrieval.signals.agreement.score = 0.4
    evidence2 = update_incremental_evidence_state(retrieval, evidence)

    assert evidence2 is evidence
    assert not np.array_equal(first_vector, evidence2.vector)
    assert evidence2.debug["retrieval_k"] == 0


def test_evidence_state_sufficiency_uses_threshold():
    retrieval = _state()
    evidence = update_incremental_evidence_state(retrieval)
    assert evidence_state_sufficient(evidence, 0.75) is True

    evidence.evidence_score = 0.50
    assert evidence_state_sufficient(evidence, 0.75) is False

