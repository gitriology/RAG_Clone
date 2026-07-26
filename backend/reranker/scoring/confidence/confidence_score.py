def compute_confidence(
    documents,
    validation,
    retrieval_confidence=0.0
):
    """
    Computes an overall confidence score using multiple signals.

    Signals:
    - reranker score
    - context relevance
    - answer validation
    - retrieval confidence

    Returns:
        float (0-1)
    """

    if not documents:
        return 0.0

    rerank_score = sum(
        doc.get("rerank_score", 0.0)
        for doc in documents
    ) / len(documents)

    context_score = sum(
        doc.get("context_score", 0.0)
        for doc in documents
    ) / len(documents)

    answer_score = validation.get(
        "confidence",
        0.0
    )

    overall = (

        0.40 * rerank_score +

        0.20 * context_score +

        0.20 * answer_score +

        0.20 * retrieval_confidence

    )

    return min(overall, 1.0)