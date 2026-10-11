"""Evidence uncertainty from signal strength, dispersion and entropy."""

from __future__ import annotations

import numpy as np

from backend.evidence_state.state.evidence_state import EvidenceUncertainty


def compute_uncertainty(evidence_state):
    uncertainty = EvidenceUncertainty()
    vector = evidence_state.weighted_vector

    if vector is None or len(vector) == 0:
        uncertainty.uncertainty_score = 1.0
        evidence_state.uncertainty = uncertainty
        evidence_state.signals.uncertainty.score = 1.0
        return evidence_state

    values = np.clip(np.abs(np.asarray(vector, dtype=float)), 0.0, 1.0)
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    strength = float(values.mean()) if values.size else 0.0

    total = float(values.sum())
    if total <= 1e-12:
        # No signal is lack of evidence, not certainty.
        uncertainty.entropy = 0.0
        uncertainty.confidence_gap = 0.0
        uncertainty.uncertainty_score = 1.0
        evidence_state.uncertainty = uncertainty
        evidence_state.signals.uncertainty.score = 1.0
        return evidence_state

    probabilities = values / total
    entropy = float(
        -np.sum(probabilities * np.log2(probabilities + 1e-12))
    )
    max_entropy = float(np.log2(max(len(values), 2)))
    normalized_entropy = max(0.0, min(1.0, entropy / max_entropy))

    ordered = np.sort(values)
    uncertainty.confidence_gap = (
        float(ordered[-1] - ordered[-2]) if len(ordered) >= 2 else 0.0
    )
    uncertainty.entropy = entropy

    # Variance is bounded by 0.25 for [0,1] values; normalize it to [0,1].
    normalized_dispersion = max(
        0.0,
        min(1.0, float(values.var()) / 0.25),
    )
    uncertainty_score = (
        0.70 * (1.0 - strength)
        + 0.20 * normalized_dispersion
        + 0.10 * normalized_entropy
    )
    uncertainty.uncertainty_score = round(
        max(0.0, min(1.0, uncertainty_score)),
        4,
    )

    evidence_state.uncertainty = uncertainty
    evidence_state.signals.uncertainty.score = uncertainty.uncertainty_score
    return evidence_state
