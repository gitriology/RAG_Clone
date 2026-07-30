from dataclasses import dataclass


@dataclass
class VectorStatistics:
    """
    Statistics describing the Evidence State Vector.
    """

    total_features: int = 0

    active_features: int = 0

    sparsity: float = 0.0

    mean_value: float = 0.0

    max_value: float = 0.0

    min_value: float = 0.0

    variance: float = 0.0

    l2_norm: float = 0.0

    weighted_mean: float = 0.0

    weighted_norm: float = 0.0

    weighted_variance: float = 0.0

    normalization_method: str = "min-max"

    normalization_min: float = 0.0

    normalization_max: float = 0.0

    normalization_range: float = 0.0