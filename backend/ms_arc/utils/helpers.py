from typing import List


def normalize(value: float,
              min_value: float,
              max_value: float) -> float:
    """
    Normalize a value into [0,1].
    """

    if max_value == min_value:
        return 0.0

    value = max(min(value, max_value), min_value)

    return (value - min_value) / (max_value - min_value)


def average(values: List[float]) -> float:
    """
    Compute arithmetic mean.
    """

    if not values:
        return 0.0

    return sum(values) / len(values)