from backend.retrieval.lexical.bm25 import search_bm25
import numpy as np


def hybrid_search(query, model, faiss_index, bm25, tokenized, texts, k=10):
    """
    Hybrid Retrieval using weighted score fusion.
    """
    # Dense Retrieval

    query_embedding = model.encode([query])

    dense_scores, dense_ids = faiss_index.search(query_embedding, k)

    dense_scores = dense_scores[0]
    dense_ids = dense_ids[0]

    # Convert FAISS distance → similarity
    dense_similarity = 1 / (1 + dense_scores)

    # Normalize Dense Scores
    if np.max(dense_similarity) != np.min(dense_similarity):
        dense_similarity = (
            dense_similarity - np.min(dense_similarity)
        ) / (
            np.max(dense_similarity) - np.min(dense_similarity)
        )
    else:
        dense_similarity = np.ones_like(dense_similarity)

    # ==================================================
    # BM25 Retrieval
    # ==================================================

    bm25_scores = bm25.get_scores(query.lower().split())

    bm25_ids = np.argsort(bm25_scores)[::-1][:k]

    bm25_top_scores = bm25_scores[bm25_ids]

    # Normalize BM25 Scores
    if np.max(bm25_top_scores) != np.min(bm25_top_scores):
        bm25_top_scores = (
            bm25_top_scores - np.min(bm25_top_scores)
        ) / (
            np.max(bm25_top_scores) - np.min(bm25_top_scores)
        )
    else:
        bm25_top_scores = np.ones_like(bm25_top_scores)
        
    # Hybrid Fusion

    dense_dict = {}
    bm25_dict = {}
    hybrid_scores = {}

    # Dense contributes 60%
    for doc_id, score in zip(dense_ids, dense_similarity):
        hybrid_scores[doc_id] = 0.6 * float(score)

    # BM25 contributes 40%
    for doc_id, score in zip(bm25_ids, bm25_top_scores):

        if doc_id in hybrid_scores:
            hybrid_scores[doc_id] += 0.4 * float(score)
        else:
            hybrid_scores[doc_id] = 0.4 * float(score)

    # ==================================================
    # Ranking
    # ==================================================

    ranked = sorted(
        hybrid_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    ranked_ids = [doc_id for doc_id, _ in ranked[:k]]

    return [texts[i] for i in ranked_ids]