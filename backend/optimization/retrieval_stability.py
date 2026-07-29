def retrieval_stability(previous_ids, current_ids):
    """
    Computes stability between two retrieval iterations.

    Returns:
        Stability score (0–1)
    """

    previous = set(previous_ids)
    current = set(current_ids)

    intersection = previous.intersection(current)
    union = previous.union(current)

    if len(union) == 0:
        return 0.0

    return round(len(intersection) / len(union), 3)