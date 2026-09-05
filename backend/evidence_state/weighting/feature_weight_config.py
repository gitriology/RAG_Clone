"""Runtime selection of EvidenceState feature weights.

Optimization #23
----------------
Manual weights remain the default for backwards compatibility. Set:

    EVIDENCE_WEIGHT_MODE=tuned

to load a validated tuned-weight file. If the file is missing, malformed, or
incomplete, the resolver safely falls back to the manual baseline.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Mapping

from backend.evidence_state.weighting.feature_weights import CANONICAL_FEATURE_WEIGHTS
from backend.evidence_state.weighting.feature_weight_tuner import load_weights


DEFAULT_TUNED_WEIGHTS_FILE = (
    Path(__file__).resolve().parent / "tuned_feature_weights.json"
)


def resolve_feature_weights(
    *,
    mode: str | None = None,
    tuned_path: str | Path | None = None,
    baseline: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    baseline = dict(baseline or CANONICAL_FEATURE_WEIGHTS)
    selected_mode = str(
        mode if mode is not None else os.getenv("EVIDENCE_WEIGHT_MODE", "manual")
    ).strip().lower()

    if selected_mode != "tuned":
        return baseline

    path = Path(
        tuned_path
        if tuned_path is not None
        else os.getenv("EVIDENCE_TUNED_WEIGHTS_FILE", DEFAULT_TUNED_WEIGHTS_FILE)
    )

    try:
        tuned = load_weights(path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return baseline

    # Keep the feature space complete. Missing tuned entries use the baseline.
    merged = {
        name: float(tuned.get(name, value))
        for name, value in baseline.items()
    }

    # A stable mean-1 scale is required by Optimization #23.
    positive = [v for v in merged.values() if v > 0]
    if not positive:
        return baseline
    mean = sum(merged.values()) / len(merged)
    if mean <= 0:
        return baseline
    return {name: value / mean for name, value in merged.items()}
