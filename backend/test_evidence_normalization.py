import numpy as np

from backend.evidence_state.normalization import normalize_features
from backend.evidence_state.state.evidence_state import EvidenceFeature, EvidenceState


def _state(*features):
    return EvidenceState(features=list(features))


def test_normalization_is_stable_across_queries():
    first = _state(
        EvidenceFeature("retrieval_confidence", 0.8, "retrieval"),
        EvidenceFeature("graph_score", 0.4, "graph"),
    )
    second = _state(
        EvidenceFeature("retrieval_confidence", 0.8, "retrieval"),
        EvidenceFeature("graph_score", 0.9, "graph"),
    )

    normalize_features(first)
    normalize_features(second)

    assert first.features[0].normalized_value == 0.8
    assert second.features[0].normalized_value == 0.8


def test_count_feature_uses_fixed_range_and_clips():
    state = _state(EvidenceFeature("recommended_topk", 5, "retrieval"))
    normalize_features(state)
    assert np.isclose(state.features[0].normalized_value, 4 / 9)

    state = _state(EvidenceFeature("recommended_topk", 50, "retrieval"))
    normalize_features(state)
    assert state.features[0].normalized_value == 1.0


def test_duplicate_feature_names_are_source_safe():
    state = _state(
        EvidenceFeature("evidence_quality", 0.75, "retrieval"),
        EvidenceFeature("evidence_quality", 0.25, "quality"),
    )
    normalize_features(state)

    assert state.features[0].normalized_value == 0.75
    assert state.features[1].normalized_value == 0.25


def test_unknown_feature_does_not_revert_to_query_local_scaling():
    state = _state(
        EvidenceFeature("new_signal", 0.8, "research"),
        EvidenceFeature("other_signal", 0.2, "research"),
    )
    normalize_features(state)

    assert state.features[0].normalized_value == 0.8
    assert state.features[1].normalized_value == 0.2
    assert state.statistics.normalization_method == "feature-specific-min-max-v1"
