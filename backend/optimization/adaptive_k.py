def choose_k(complexity):
    """
    Adaptive retrieval depth.
    """

    mapping = {
        "simple": 3,
        "medium": 5,
        "complex": 8
    }

    return mapping.get(complexity, 5)