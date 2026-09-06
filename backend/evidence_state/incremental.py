"""Incremental, retrieval-stage Evidence State updates (Optimization #36).

The full EvidenceState pipeline is intentionally kept for the final evidence
stage.  During MS-ARC retrieval iterations we only update the cheap retrieval
features needed for a sufficiency decision.  This avoids rebuilding graph,
quality, ranking and diagnostic feature state after every +5 retrieval step.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from backend.evidence_state.state.evidence_state import EvidenceState
from backend.evidence_state.features.retrieval_features import (
    extract_retrieval_features,
)
from backend.evidence_state.normalization import normalize_features
from backend.evidence_state.weighting.feature_weights import apply_feature_weights
from backend.evidence_state.weighting.weighted_vector import build_weighted_vector
from backend.evidence_state.vector_builder import build_vector, update_weighted_statistics
from backend.evidence_state.diagnostics.vector_health import compute_vector_health
from backend.evidence_state.diagnostics.uncertainty import compute_uncertainty
from backend.evidence_state.weighting.evidence_score import compute_evidence_score


RETRIEVAL_STAGE_FEATURES = {
    "agreement",
    "margin",
    "stability",
    "retrieval_confidence",
    "query_complexity",
    "recommended_topk",
    "novelty",
    "evidence_quality",
    "evidence_coverage",
    "evidence_diversity",
    "evidence_consistency",
}


def update_incremental_evidence_state(
    retrieval_state,
    evidence_state: Optional[EvidenceState] = None,
) -> EvidenceState:
    """Update only retrieval-stage EvidenceState after one MS-ARC iteration.

    The operation deliberately excludes graph construction, graph analytics,
    quality/ranking extraction and profile/reasoning stages.  Vector health,
    uncertainty and the weighted evidence score are cheap operations over the
    small retrieval feature vector, so they are refreshed for the controller.
    """

    if retrieval_state is None:
        raise ValueError("retrieval_state must not be None")

    state = evidence_state or EvidenceState()

    features = extract_retrieval_features(retrieval_state)
    state.features = [
        feature
        for feature in features
        if feature.name in RETRIEVAL_STAGE_FEATURES
    ]
    state.feature_lookup = {
        feature.name: feature
        for feature in state.features
    }

    # Reuse the normal, stable feature-specific transform used by the full
    # EvidenceState pipeline; no query-local min/max is introduced here.
    normalize_features(state)
    apply_feature_weights(state)
    build_vector(state)
    build_weighted_vector(state)
    update_weighted_statistics(state)
    compute_vector_health(state)
    compute_uncertainty(state)
    compute_evidence_score(state)

    state.debug = getattr(state, "debug", {})
    state.debug["update_mode"] = "incremental_retrieval"
    state.debug["feature_count"] = len(state.features)
    state.debug["retrieval_k"] = len(retrieval_state.merged_results)

    return state


def evidence_state_sufficient(
    evidence_state: EvidenceState,
    threshold: float = 0.75,
) -> bool:
    """Return whether the cheap retrieval-stage EvidenceState is sufficient."""

    if evidence_state is None:
        return False

    return float(evidence_state.evidence_score) >= float(threshold)
