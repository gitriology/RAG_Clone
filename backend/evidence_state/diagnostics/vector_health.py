"""Evidence-vector health metrics.

Health measures the quality of evidence signals, not how difficult a query is
or how many documents the adaptive controller requested.
"""

from __future__ import annotations

import numpy as np

from backend.evidence_state.state.evidence_state import VectorHealth


NON_QUALITY_FEATURES = {"query_complexity", "recommended_topk"}


def compute_vector_health(evidence_state):
    features = [
        feature
        for feature in (evidence_state.features or [])
        if feature.name not in NON_QUALITY_FEATURES
    ]
    health = VectorHealth()

    if not features:
        evidence_state.health = health
        evidence_state.signals.health.score = 0.0
        return evidence_state

    values = np.asarray(
        [
            max(0.0, min(1.0, float(feature.normalized_value)))
            if np.isfinite(float(feature.normalized_value))
            else 0.0
            for feature in features
        ],
        dtype=float,
    )

    strength = float(values.mean()) if values.size else 0.0
    maximum = float(values.max()) if values.size else 0.0

    health.completeness = float(np.count_nonzero(values > 1e-8) / len(values))

    # Since values are clipped to [0, 1], population variance is at most 0.25.
    # Scale that full range to [0, 1] rather than using 1 - variance, which
    # would incorrectly keep consistency >= 0.75 for every possible vector.
    variance = float(values.var()) if values.size else 0.0
    health.consistency = max(0.0, 1.0 - min(1.0, variance / 0.25))

    health.balance = strength / maximum if maximum > 1e-8 else 0.0

    # Strength gates structural quality: a uniformly tiny vector must not be
    # declared healthy merely because it is dense, balanced and low-variance.
    structural_quality = (
        0.40 * health.completeness
        + 0.30 * health.consistency
        + 0.30 * health.balance
    )
    health.health_score = round(
        max(0.0, min(1.0, strength * structural_quality)),
        4,
    )

    evidence_state.health = health
    evidence_state.signals.health.score = health.health_score
    return evidence_state
