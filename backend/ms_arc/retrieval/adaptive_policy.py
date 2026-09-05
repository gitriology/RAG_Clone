"""Cost-aware retrieval policy scaffold for MS-ARC (Optimization #26).

This module makes the adaptive architecture explicit without changing the
production retrieval loop yet. Optimization #28 will activate the iterative
retrieve -> evaluate -> retrieve-more loop using this policy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveRetrievalPlan:
    initial_k: int
    expansion_step: int
    max_k: int
    confidence_threshold: float
    convergence_epsilon: float


class AdaptiveRetrievalPolicy:
    """Build a deterministic retrieval plan from query complexity.

    #26 separates policy from execution. That keeps the controller testable
    and lets #28 add iterative retrieval without rewriting the rest of MS-ARC.
    """

    def __init__(
        self,
        initial_k: int = 5,
        expansion_step: int = 5,
        max_k: int = 20,
        confidence_threshold: float = 0.75,
        convergence_epsilon: float = 0.02,
    ):
        if initial_k < 1 or expansion_step < 1 or max_k < initial_k:
            raise ValueError("Invalid adaptive retrieval policy bounds")
        self.initial_k = int(initial_k)
        self.expansion_step = int(expansion_step)
        self.max_k = int(max_k)
        self.confidence_threshold = float(confidence_threshold)
        self.convergence_epsilon = float(convergence_epsilon)

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, value))

    def plan(self, query_complexity: float, recommended_topk: int) -> AdaptiveRetrievalPlan:
        """Return initial depth plus controlled expansion bounds.

        Complexity influences the ceiling, while the first pass stays small
        (Top-5 by default) as required by the adaptive architecture.
        """
        complexity = self._clamp(float(query_complexity))
        requested = max(1, int(recommended_topk))

        complexity_ceiling = self.initial_k + round(complexity * (self.max_k - self.initial_k))
        target_max = max(self.initial_k, requested, complexity_ceiling)
        target_max = min(self.max_k, target_max)

        return AdaptiveRetrievalPlan(
            initial_k=min(self.initial_k, target_max),
            expansion_step=self.expansion_step,
            max_k=target_max,
            confidence_threshold=self.confidence_threshold,
            convergence_epsilon=self.convergence_epsilon,
        )

    def should_stop(
        self,
        confidence: float,
        previous_confidence: float | None = None,
        current_k: int | None = None,
        plan: AdaptiveRetrievalPlan | None = None,
    ) -> bool:
        """Cheap sufficiency/convergence gate for the future iterative loop."""
        if confidence >= self.confidence_threshold:
            return True
        if previous_confidence is not None:
            gain = float(confidence) - float(previous_confidence)
            if gain <= self.convergence_epsilon:
                return True
        if plan is not None and current_k is not None and current_k >= plan.max_k:
            return True
        return False
