"""Data-driven feature-weight tuning for EvidenceState.

Optimization #23
----------------
Learns positive feature weights from a labelled validation set while keeping
the existing manual weights as a stable prior. Learned weights are normalized
to mean 1.0, which preserves interpretability and prevents arbitrary scale
changes from affecting the downstream weighted-vector mean.

Validation format
-----------------
A JSON list of records:

[
  {
    "label": 1,
    "features": {
      "retrieval_confidence": 0.9,
      "graph_score": 0.8
    }
  },
  ...
]

Feature values should use the same normalized [0, 1] representation consumed
by EvidenceState. Missing features are treated as 0.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def _sigmoid(x: float) -> float:
    x = max(-40.0, min(40.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def load_validation_dataset(path: str | Path) -> List[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list) or not data:
        raise ValueError("Validation dataset must be a non-empty JSON list.")

    validated: List[dict] = []
    for index, row in enumerate(data, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Validation row {index} must be an object.")
        label = row.get("label")
        features = row.get("features")
        if label not in (0, 1, 0.0, 1.0, False, True):
            raise ValueError(f"Validation row {index} has invalid label: {label!r}.")
        if not isinstance(features, dict):
            raise ValueError(f"Validation row {index} must contain a 'features' object.")
        clean = {}
        for name, value in features.items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Validation row {index} feature {name!r} is not numeric."
                )
            if not math.isfinite(value):
                raise ValueError(
                    f"Validation row {index} feature {name!r} is not finite."
                )
            clean[str(name)] = max(0.0, min(1.0, value))
        validated.append({"label": int(bool(label)), "features": clean})

    if len({row["label"] for row in validated}) < 2:
        raise ValueError("Validation dataset must contain both positive and negative labels.")

    return validated


def _matrix(
    dataset: Sequence[Mapping[str, object]],
    feature_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.zeros((len(dataset), len(feature_names)), dtype=float)
    y = np.zeros(len(dataset), dtype=float)
    index = {name: i for i, name in enumerate(feature_names)}

    for row_idx, row in enumerate(dataset):
        y[row_idx] = float(row["label"])
        features = row["features"]
        for name, value in features.items():
            if name in index:
                x[row_idx, index[name]] = float(value)

    return x, y


WEIGHT_MIN = 0.25
WEIGHT_MAX = 2.50


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Keep learned weights positive, bounded, and mean-normalized."""

    weights = np.maximum(np.asarray(weights, dtype=float), 1e-8)
    mean = float(weights.mean())
    if mean <= 0:
        return np.ones_like(weights)
    weights = weights / mean

    # A bounded ratio prevents one feature from dominating the EvidenceState
    # vector simply because the validation set is small.
    for _ in range(8):
        weights = np.clip(weights, WEIGHT_MIN, WEIGHT_MAX)
        weights = weights / float(weights.mean())
        if np.all(weights >= WEIGHT_MIN - 1e-9) and np.all(weights <= WEIGHT_MAX + 1e-9):
            break
    return weights


def _loss_and_gradient(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    prior_theta: np.ndarray,
    l2: float,
) -> Tuple[float, np.ndarray]:
    raw_weights = np.exp(np.clip(theta, -8.0, 8.0))
    weights = _normalize_weights(raw_weights)
    scores = (x @ weights) / max(float(weights.sum()), 1e-8)
    # EvidenceState ultimately uses the weighted-vector mean. Center the
    # learning objective on the natural [0, 1] midpoint rather than the raw
    # sum, otherwise adding more features would artificially inflate scores.
    probabilities = np.array([_sigmoid((v - 0.5) * 10.0) for v in scores])

    eps = 1e-9
    loss = -float(
        np.mean(
            y * np.log(probabilities + eps)
            + (1.0 - y) * np.log(1.0 - probabilities + eps)
        )
    )
    loss += l2 * float(np.mean((theta - prior_theta) ** 2))

    residual = probabilities - y
    grad_w = (x.T @ residual) / max(len(y), 1)
    # Jacobian of mean-normalized positive weights.
    mean_raw = float(raw_weights.mean())
    dwdtheta = np.diag(raw_weights / mean_raw) - np.outer(
        raw_weights, raw_weights
    ) / (len(raw_weights) * mean_raw**2)
    grad_theta = dwdtheta @ grad_w
    grad_theta += 2.0 * l2 * (theta - prior_theta) / max(len(theta), 1)
    return loss, grad_theta


def tune_feature_weights(
    dataset: Sequence[Mapping[str, object]],
    baseline_weights: Mapping[str, float],
    *,
    feature_names: Iterable[str] | None = None,
    learning_rate: float = 0.08,
    epochs: int = 1500,
    l2: float = 0.20,
) -> Dict[str, float]:
    """Tune positive weights and return a mean-1 normalized mapping."""

    names = list(feature_names or baseline_weights.keys())
    if not names:
        raise ValueError("No feature names supplied for tuning.")

    x, y = _matrix(dataset, names)
    baseline = np.array(
        [max(float(baseline_weights.get(name, 1.0)), 1e-8) for name in names],
        dtype=float,
    )
    prior_theta = np.log(baseline / float(baseline.mean()))
    theta = prior_theta.copy()

    # Deterministic full-batch Adam-style updates.
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    beta1, beta2 = 0.9, 0.999

    for step in range(1, max(int(epochs), 1) + 1):
        _, grad = _loss_and_gradient(theta, x, y, prior_theta, l2)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        theta -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        theta = np.clip(theta, -6.0, 6.0)

    tuned = _normalize_weights(np.exp(theta))
    return {name: round(float(weight), 6) for name, weight in zip(names, tuned)}


def save_weights(
    weights: Mapping[str, float],
    path: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Persist tuned weights and provenance in a stable JSON format."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "optimization": "Optimization #23",
        "method": "positive-logistic-weight-tuning",
        "normalization": "mean-weight=1.0",
        "weights": {str(k): float(v) for k, v in weights.items()},
        "metadata": dict(metadata or {}),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=False)


def load_weights(path: str | Path) -> Dict[str, float]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    weights = payload.get("weights") if isinstance(payload, dict) else None
    if not isinstance(weights, dict):
        raise ValueError("Tuned weight file must contain a 'weights' object.")
    result = {}
    for name, value in weights.items():
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            result[str(name)] = value
    if not result:
        raise ValueError("Tuned weight file contains no valid positive weights.")
    return result


def predict_with_weights(
    dataset: Sequence[Mapping[str, object]],
    weights: Mapping[str, float],
    *,
    threshold: float = 0.5,
) -> List[int]:
    names = list(weights)
    x, _ = _matrix(dataset, names)
    normalized = _normalize_weights(np.array([weights[n] for n in names]))
    scores = (x @ normalized) / max(float(normalized.sum()), 1e-8)
    return [int(float(score) >= threshold) for score in scores]


def accuracy(
    dataset: Sequence[Mapping[str, object]],
    weights: Mapping[str, float],
) -> float:
    predictions = predict_with_weights(dataset, weights)
    labels = [int(row["label"]) for row in dataset]
    return float(np.mean(np.asarray(predictions) == np.asarray(labels)))


def compare_weight_sets(
    dataset: Sequence[Mapping[str, object]],
    *,
    equal_weights: Mapping[str, float],
    manual_weights: Mapping[str, float],
    tuned_weights: Mapping[str, float],
) -> Dict[str, float]:
    return {
        "equal": round(accuracy(dataset, equal_weights), 4),
        "manual": round(accuracy(dataset, manual_weights), 4),
        "tuned": round(accuracy(dataset, tuned_weights), 4),
    }
