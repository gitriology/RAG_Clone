def retrieval_agreement(dense_ids, bm25_ids):
    """
    Computes overlap between Dense Retrieval and BM25.

    Returns:
        Agreement score between 0 and 1.
    """

    dense_set = set(dense_ids)
    bm25_set = set(bm25_ids)

    intersection = dense_set.intersection(bm25_set)
    union = dense_set.union(bm25_set)

    if len(union) == 0:
        return 0.0

    return round(len(intersection) / len(union), 3)