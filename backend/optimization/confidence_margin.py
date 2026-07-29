def confidence_margin(ranked_docs):
    """
    Computes confidence margin between
    top-1 and top-2 reranked documents.
    """

    if len(ranked_docs) < 2:
        return 1.0

    top1 = ranked_docs[0]["rerank_score"]
    top2 = ranked_docs[1]["rerank_score"]

    return round(top1 - top2, 3)