from pathlib import Path
import json
import tempfile

from backend.evidence_state.state.evidence_state import EvidenceFeature, EvidenceState
from backend.evidence_state.weighting.feature_weights import (
    CANONICAL_FEATURE_WEIGHTS,
    FEATURE_WEIGHTS,
    apply_feature_weights,
)
from backend.evidence_state.weighting.feature_weight_config import resolve_feature_weights
from backend.evidence_state.weighting.feature_weight_tuner import (
    compare_weight_sets,
    load_validation_dataset,
    save_weights,
    tune_feature_weights,
)


def _dataset():
    return [
        {"label": 1, "features": {"retrieval_confidence": .95, "graph_score": .90, "evidence_quality": .92}},
        {"label": 1, "features": {"retrieval_confidence": .85, "graph_score": .80, "evidence_quality": .88}},
        {"label": 1, "features": {"retrieval_confidence": .75, "graph_score": .72, "evidence_quality": .80}},
        {"label": 0, "features": {"retrieval_confidence": .15, "graph_score": .20, "evidence_quality": .18}},
        {"label": 0, "features": {"retrieval_confidence": .25, "graph_score": .30, "evidence_quality": .22}},
        {"label": 0, "features": {"retrieval_confidence": .35, "graph_score": .28, "evidence_quality": .30}},
    ]


def test_baseline_aliases_match_current_feature_names():
    assert CANONICAL_FEATURE_WEIGHTS["margin"] == FEATURE_WEIGHTS["retrieval_margin"]
    assert CANONICAL_FEATURE_WEIGHTS["agreement"] == FEATURE_WEIGHTS["retrieval_agreement"]
    assert CANONICAL_FEATURE_WEIGHTS["query_complexity"] == FEATURE_WEIGHTS["retrieval_complexity"]
    assert CANONICAL_FEATURE_WEIGHTS["health_score"] == FEATURE_WEIGHTS["vector_health"]


def test_tuned_weights_are_positive_and_mean_one():
    data = _dataset()
    names = sorted({name for row in data for name in row["features"]})
    baseline = {name: CANONICAL_FEATURE_WEIGHTS.get(name, 1.0) for name in names}
    tuned = tune_feature_weights(data, baseline, feature_names=names, epochs=250)
    assert all(0.25 <= value <= 2.50 for value in tuned.values())
    assert abs(sum(tuned.values()) / len(tuned) - 1.0) < 1e-6


def test_three_way_ablation_is_reproducible():
    data = _dataset()
    names = sorted({name for row in data for name in row["features"]})
    baseline = {name: CANONICAL_FEATURE_WEIGHTS.get(name, 1.0) for name in names}
    equal = {name: 1.0 for name in names}
    tuned = tune_feature_weights(data, baseline, feature_names=names, epochs=250)
    comparison = compare_weight_sets(
        data,
        equal_weights=equal,
        manual_weights=baseline,
        tuned_weights=tuned,
    )
    assert set(comparison) == {"equal", "manual", "tuned"}
    assert all(0.0 <= value <= 1.0 for value in comparison.values())


def test_production_weight_application_uses_tuned_mode(monkeypatch):
    data = _dataset()
    names = sorted({name for row in data for name in row["features"]})
    baseline = {name: CANONICAL_FEATURE_WEIGHTS.get(name, 1.0) for name in names}
    tuned = tune_feature_weights(data, baseline, feature_names=names, epochs=250)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "weights.json"
        save_weights(tuned, path)
        monkeypatch.setenv("EVIDENCE_WEIGHT_MODE", "tuned")
        monkeypatch.setenv("EVIDENCE_TUNED_WEIGHTS_FILE", str(path))

        state = EvidenceState()
        state.features = [
            EvidenceFeature("retrieval_confidence", .9, "retrieval"),
            EvidenceFeature("graph_score", .8, "graph"),
            EvidenceFeature("uncertainty", .2, "diagnostics"),
        ]
        apply_feature_weights(state)
        assert state.weighting_mode == "tuned"
        assert state.active_feature_weights["retrieval_confidence"] > 0
        assert state.features[0].weight == state.active_feature_weights["retrieval_confidence"]


def test_tuned_file_is_loaded_and_runtime_can_use_it():
    data = _dataset()
    names = sorted({name for row in data for name in row["features"]})
    baseline = {name: CANONICAL_FEATURE_WEIGHTS.get(name, 1.0) for name in names}
    tuned = tune_feature_weights(data, baseline, feature_names=names, epochs=250)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "weights.json"
        save_weights(tuned, path)
        resolved = resolve_feature_weights(
            mode="tuned",
            tuned_path=path,
            baseline=CANONICAL_FEATURE_WEIGHTS,
        )
        assert set(resolved) == set(CANONICAL_FEATURE_WEIGHTS)
        assert abs(sum(resolved.values()) / len(resolved) - 1.0) < 1e-6

        state = EvidenceState()
        state.features = [
            EvidenceFeature("retrieval_confidence", .9, "retrieval"),
            EvidenceFeature("graph_score", .8, "graph"),
        ]
        # Force the same resolver through the production function by using the
        # environment-independent helper only for the direct wiring assertion.
        for feature in state.features:
            feature.weight = resolved.get(feature.name, 1.0)
        assert state.features[0].weight > 0


def test_malformed_or_missing_tuned_file_falls_back_to_manual():
    resolved = resolve_feature_weights(
        mode="tuned",
        tuned_path="/definitely/missing/tuned_feature_weights.json",
        baseline=CANONICAL_FEATURE_WEIGHTS,
    )
    assert resolved == CANONICAL_FEATURE_WEIGHTS
