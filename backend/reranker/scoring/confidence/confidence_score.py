import math


def compute_confidence(
    documents,
    validation,
    retrieval_confidence=0.0
):
    """
    Computes an overall confidence score using multiple signals.

    Signals:
    - reranker score (CrossEncoder logits -> sigmoid probability)
    - context relevance
    - answer validation
    - retrieval confidence

    Returns:
        float (0-1)
    """

    if not documents:
        return 0.0

    # =====================================================
    # Normalize CrossEncoder logits using Sigmoid
    # =====================================================

    rerank_scores = []

    for doc in documents:

        logit = doc.get("rerank_score", 0.0)

        probability = 1 / (1 + math.exp(-logit))

        rerank_scores.append(probability)

    rerank_score = sum(rerank_scores) / len(rerank_scores)

    # =====================================================
    # Context Score
    # =====================================================

    context_score = sum(
        doc.get("context_score", 0.0)
        for doc in documents
    ) / len(documents)

    # =====================================================
    # Answer Validation Score
    # =====================================================

    answer_score = validation.get(
        "confidence",
        0.0
    )

    # Convert NumPy types to Python float if needed
    answer_score = float(answer_score)

    # =====================================================
    # Overall Confidence
    # =====================================================

    overall = (

        0.40 * rerank_score +

        0.20 * context_score +

        0.20 * answer_score +

        0.20 * retrieval_confidence

    )

    # Clamp to [0, 1]
    overall = max(0.0, min(overall, 1.0))

    # =====================================================
    # Debug Output
    # =====================================================

    print("\n" + "=" * 60)
    print("Pipeline Confidence Debug")
    print("=" * 60)

    print(f"Average Rerank Score     : {rerank_score:.4f}")
    print(f"Average Context Score    : {context_score:.4f}")
    print(f"Answer Confidence        : {answer_score:.4f}")
    print(f"Retrieval Confidence     : {retrieval_confidence:.4f}")

    print("-" * 60)

    print(f"Final Pipeline Confidence: {overall:.4f}")

    print("=" * 60)

    return overall