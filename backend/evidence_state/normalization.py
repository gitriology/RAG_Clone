"""Feature-specific, stable normalization for EvidenceState.

Optimization #22
----------------
The previous implementation normalized every feature against the minimum and
maximum values *observed in the current query*.  That makes a normalized value
context-dependent: the same raw value can map to a different vector value on a
subsequent query.

This module instead uses stable, feature-specific domains.  Bounded confidence
signals keep their natural [0, 1] meaning, while count-like signals use fixed
operational ranges.  Values outside a configured range are clipped rather than
allowed to distort the scale for the current query.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


# (minimum, maximum) for each feature.  Most EvidenceState signals are already
# probabilities/scores in [0, 1].  Count-like features get an explicit stable
# operational domain so their meaning does not depend on the other features in
# the same query.
DEFAULT_FEATURE_RANGES: Dict[str, Tuple[float, float]] = {
    # Retrieval
    "agreement": (0.0, 1.0),
    "margin": (0.0, 1.0),
    "stability": (0.0, 1.0),
    "retrieval_confidence": (0.0, 1.0),
    "query_complexity": (0.0, 1.0),
    "recommended_topk": (1.0, 10.0),
    "novelty": (0.0, 1.0),
    "evidence_coverage": (0.0, 1.0),
    "evidence_diversity": (0.0, 1.0),
    "evidence_consistency": (0.0, 1.0),

    # Evidence quality / graph quality
    "evidence_quality": (0.0, 1.0),
    "graph_score": (0.0, 1.0),
    "graph_density": (0.0, 1.0),
    "graph_coherence": (0.0, 1.0),
    "graph_consensus": (0.0, 1.0),

    # Graph topology / centrality. These are intentionally bounded so a larger
    # retrieval depth cannot redefine the scale of an individual query.
    "connected_components": (0.0, 20.0),
    "average_degree": (0.0, 20.0),
    "average_betweenness": (0.0, 1.0),
    "average_closeness": (0.0, 1.0),
    "average_eigenvector": (0.0, 1.0),
    "average_pagerank": (0.0, 1.0),

    # Ranking
    "ranking_average_score": (0.0, 1.0),
    "ranking_highest_score": (0.0, 1.0),
    "ranking_lowest_score": (0.0, 1.0),

    # Diagnostics (if/when refreshed into the state)
    "health_score": (0.0, 1.0),
    "uncertainty": (0.0, 1.0),
    "vector_sparsity": (0.0, 1.0),
    "vector_norm": (0.0, 10.0),
}

# Some names (notably evidence_quality) occur in more than one feature source.
# The source-specific overrides make those features independently addressable.
SOURCE_FEATURE_RANGES: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("retrieval", "evidence_quality"): (0.0, 1.0),
    ("quality", "evidence_quality"): (0.0, 1.0),
}

NORMALIZATION_METHOD = "feature-specific-min-max-v1"


def _range_for(feature) -> Tuple[float, float]:
    """Return the stable normalization range for an EvidenceFeature."""

    key = (str(feature.source), str(feature.name))
    if key in SOURCE_FEATURE_RANGES:
        return SOURCE_FEATURE_RANGES[key]

    if feature.name in DEFAULT_FEATURE_RANGES:
        return DEFAULT_FEATURE_RANGES[feature.name]

    # Fail safe for a newly introduced bounded score. We deliberately do not
    # fall back to query-local min/max because that would reintroduce the old
    # instability. An explicit [0, 1] domain is the safest default for an
    # unknown EvidenceState signal.
    return (0.0, 1.0)


def normalize_features(evidence_state):
    """Normalize every feature against its stable, feature-specific domain.

    Raw values remain untouched. Normalized values are stored on each feature.
    ``normalization_min/max`` retain aggregate raw extrema for backwards
    compatibility with existing diagnostics/serialization consumers; they no
    longer define the actual normalization transform.
    """

    if not evidence_state.features:
        return evidence_state

    raw_values = np.asarray(
        [feature.value for feature in evidence_state.features],
        dtype=float,
    )

    normalized = []
    for feature in evidence_state.features:
        minimum, maximum = _range_for(feature)

        value = float(feature.value)
        if not np.isfinite(value):
            value = minimum

        clipped = min(max(value, minimum), maximum)

        if maximum == minimum:
            norm = 0.0
        else:
            norm = (clipped - minimum) / (maximum - minimum)

        feature.normalized_value = float(norm)
        normalized.append(norm)

    # Preserve existing aggregate statistics fields for compatibility. They
    # are diagnostics only under the new scheme.
    stats = evidence_state.statistics
    stats.normalization_method = NORMALIZATION_METHOD
    stats.normalization_min = float(raw_values.min())
    stats.normalization_max = float(raw_values.max())
    stats.normalization_range = float(raw_values.max() - raw_values.min())

    return evidence_state
