"""Baseline and runtime feature weights.

Optimization #23
----------------
The original manual mapping is preserved exactly as the baseline.  A small
canonical-name layer maps legacy/manual names to the feature names currently
emitted by EvidenceState, so tuning operates on the real vector dimensions.
"""

from __future__ import annotations

from typing import Dict

# Original manual baseline (kept intact for ablation/reproducibility).
FEATURE_WEIGHTS: Dict[str, float] = {
    "retrieval_confidence": 1.20,
    "retrieval_margin": 1.15,
    "retrieval_agreement": 1.10,
    "retrieval_complexity": 0.90,
    "graph_score": 1.25,
    "graph_density": 0.90,
    "graph_coherence": 1.20,
    "graph_consensus": 1.15,
    "average_degree": 0.80,
    "average_betweenness": 0.90,
    "average_closeness": 0.90,
    "average_eigenvector": 1.00,
    "average_pagerank": 1.10,
    "evidence_quality": 1.30,
    "vector_health": 1.00,
    "uncertainty": 1.10,
}

# Names used by the current EvidenceFeature extractors.
FEATURE_NAME_ALIASES = {
    "retrieval_margin": "margin",
    "retrieval_agreement": "agreement",
    "retrieval_complexity": "query_complexity",
    "vector_health": "health_score",
}

CANONICAL_FEATURE_WEIGHTS: Dict[str, float] = {}
for _name, _weight in FEATURE_WEIGHTS.items():
    _canonical = FEATURE_NAME_ALIASES.get(_name, _name)
    # If two legacy names ever map to one dimension, keep the stronger prior.
    CANONICAL_FEATURE_WEIGHTS[_canonical] = max(
        float(_weight),
        CANONICAL_FEATURE_WEIGHTS.get(_canonical, 0.0),
    )


def get_active_feature_weights() -> Dict[str, float]:
    """Resolve the manual or tuned mapping selected for this process."""

    from backend.evidence_state.weighting.feature_weight_config import (
        resolve_feature_weights,
    )

    return resolve_feature_weights(
        baseline=CANONICAL_FEATURE_WEIGHTS
    )


def apply_feature_weights(evidence_state):
    """Apply active weights to every emitted EvidenceFeature."""

    active = get_active_feature_weights()
    evidence_state.weighting_mode = (
        "tuned"
        if any(
            abs(active.get(k, 1.0) - v) > 1e-9
            for k, v in CANONICAL_FEATURE_WEIGHTS.items()
        )
        else "manual"
    )
    evidence_state.active_feature_weights = dict(active)

    for feature in evidence_state.features:
        feature.weight = float(active.get(feature.name, 1.0))

    return evidence_state
