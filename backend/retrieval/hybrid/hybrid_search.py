import numpy as np

from backend.retrieval.dense.embedder import encode_query


def _normalize(scores):
    """
    Normalize scores to [0,1]
    """

    scores = np.asarray(scores, dtype=np.float32)

    if len(scores) == 0:
        return scores

    minimum = np.min(scores)
    maximum = np.max(scores)

    if maximum == minimum:
        return np.ones_like(scores)

    return (scores - minimum) / (maximum - minimum)


def hybrid_search(
    query,
    faiss_index,
    bm25,
    texts,
    k=10,
    dense_weight=0.6,
    sparse_weight=0.4,
):
    """
    Hybrid Retrieval

    Returns

    {
        "dense_results": [...candidate pool...],
        "sparse_results": [...candidate pool...],
        "merged_results": [...top-k...]
    }
    """

    # =====================================================
    # Retrieve a larger candidate pool
    # =====================================================

    candidate_k = max(k * 4, 20)

    # =====================================================
    # Dense Retrieval
    # =====================================================

    query_embedding = encode_query(query)

    dense_scores, dense_ids = faiss_index.search(
        query_embedding,
        candidate_k
    )

    dense_scores = _normalize(dense_scores[0])
    dense_ids = dense_ids[0]

    dense_results = []
    dense_lookup = {}

    for doc_id, score in zip(dense_ids, dense_scores):

        doc = {

            "doc_id": int(doc_id),

            "text": texts[int(doc_id)],

            "dense_score": float(score),

            "bm25_score": 0.0,

            "hybrid_score": 0.0

        }

        dense_results.append(doc)

        dense_lookup[int(doc_id)] = doc

    # =====================================================
    # Sparse Retrieval
    # =====================================================

    bm25_scores = bm25.get_scores(
        query.lower().split()
    )

    bm25_ids = np.argsort(
        bm25_scores
    )[::-1][:candidate_k]

    bm25_top_scores = _normalize(
        bm25_scores[bm25_ids]
    )

    sparse_results = []
    sparse_lookup = {}

    for doc_id, score in zip(
        bm25_ids,
        bm25_top_scores
    ):

        doc = {

            "doc_id": int(doc_id),

            "text": texts[int(doc_id)],

            "dense_score": 0.0,

            "bm25_score": float(score),

            "hybrid_score": 0.0

        }

        sparse_results.append(doc)

        sparse_lookup[int(doc_id)] = doc

    # =====================================================
    # Hybrid Fusion
    # =====================================================

    merged_lookup = {}

    all_doc_ids = set(dense_lookup.keys())

    all_doc_ids.update(
        sparse_lookup.keys()
    )

    for doc_id in all_doc_ids:

        dense_score = dense_lookup.get(
            doc_id,
            {}
        ).get(
            "dense_score",
            0.0
        )

        sparse_score = sparse_lookup.get(
            doc_id,
            {}
        ).get(
            "bm25_score",
            0.0
        )

        hybrid_score = (

            dense_weight * dense_score +

            sparse_weight * sparse_score

        )

        merged_lookup[doc_id] = {

            "doc_id": doc_id,

            "text": texts[doc_id],

            "dense_score": dense_score,

            "bm25_score": sparse_score,

            "hybrid_score": hybrid_score

        }

    merged_results = sorted(

        merged_lookup.values(),

        key=lambda x: x["hybrid_score"],

        reverse=True

    )

    # Only return final Top-K after fusion
    merged_results = merged_results[:k]

    return {

        "dense_results": dense_results,

        "sparse_results": sparse_results,

        "merged_results": merged_results

    }