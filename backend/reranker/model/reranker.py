from sentence_transformers import CrossEncoder

# Load model (once)
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query, documents):
    # Safety check (avoid crash)
    if not documents:
        return []

    # Create query-document pairs
    pairs = [(query, doc["text"]) for doc in documents]

    # Get relevance scores
    scores = model.predict(pairs)

    # Attach scores to documents
    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)

    # Sort documents by score (descending)
    ranked = sorted(
        documents,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return ranked