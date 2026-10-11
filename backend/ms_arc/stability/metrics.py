"""Pure ranking-overlap metrics used by adaptive retrieval."""


def jaccard(a, b):
    """Set overlap in [0, 1], with two empty rankings considered equal."""
    left, right = set(a), set(b)
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def prefix_jaccard(previous, current):
    """Compare a previous top-k against the same-sized prefix of a larger list."""
    previous = list(previous)
    current_prefix = list(current)[:len(previous)]
    return jaccard(previous, current_prefix)
