def compute_confidence(documents, validation_result):
    """
    Combines multiple signals to produce final confidence score
    """

    if not documents:
        return 0.0

    # Average rerank score
    rerank_scores = [doc.get("rerank_score", 0) for doc in documents]
    avg_rerank = sum(rerank_scores) / len(rerank_scores)

    # Normalize rerank (rough normalization)
    normalized_rerank = max(0, min(1, (avg_rerank + 3) / 6))

    # Average context score
    context_scores = [doc.get("context_score", 0) for doc in documents]
    avg_context = sum(context_scores) / len(context_scores)

    # Answer validation confidence
    answer_conf = validation_result.get("confidence", 0)

    # Weighted combination
    final_score = (
        0.4 * normalized_rerank +
        0.3 * avg_context +
        0.3 * answer_conf
    )

    return round(final_score, 3)