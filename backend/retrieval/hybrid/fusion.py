"""
Optimization #6
Hybrid Fusion Methods

Provides:

    1. Min-Max weighted fusion
    2. Reciprocal Rank Fusion (RRF)

Both methods operate on the SAME dense and BM25 candidate
lists so that the benchmark isolates the effect of the
fusion strategy.

Min-Max:
    0.6 * normalized_dense_score
    +
    0.4 * normalized_bm25_score

RRF:
    sum(1 / (rrf_k + rank))

Important:
    RRF does NOT use raw retrieval scores.
    It uses the rank produced by each retriever.
"""

from typing import Dict, List, Any


# ==========================================================
# CONFIGURATION
# ==========================================================

DEFAULT_DENSE_WEIGHT = 0.6
DEFAULT_BM25_WEIGHT = 0.4

DEFAULT_RRF_K = 60


# ==========================================================
# HELPERS
# ==========================================================

def _get_doc_id(document: Dict[str, Any]):
    """
    Extract document ID from a retrieval result.
    """

    doc_id = document.get("doc_id")

    if doc_id is None:
        doc_id = document.get("id")

    return doc_id


def _get_score(document: Dict[str, Any]):
    """
    Extract retrieval score.

    Supports common score field names.
    """

    if "score" in document:
        return float(document["score"])

    if "similarity" in document:
        return float(document["similarity"])

    if "dense_score" in document:
        return float(document["dense_score"])

    if "bm25_score" in document:
        return float(document["bm25_score"])

    return 0.0


def _normalize_scores(
    scores: List[float],
) -> List[float]:
    """
    Query-local Min-Max normalization.

    This intentionally preserves the current baseline
    behaviour because Optimization #6 is an ablation
    comparing this baseline against RRF.

    Formula:

        (x - min) / (max - min)

    If all scores are equal, every score becomes 1.0.
    """

    if not scores:
        return []

    minimum = min(scores)
    maximum = max(scores)

    if maximum == minimum:
        return [1.0 for _ in scores]

    return [
        (score - minimum) / (maximum - minimum)
        for score in scores
    ]


# ==========================================================
# PREPARE RANKINGS
# ==========================================================

def prepare_ranking(
    documents: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Converts raw retrieval results into a consistent format.

    Output:

        {
            "doc_id": ...,
            "rank": ...,
            "score": ...,
            "document": ...
        }
    """

    ranking = []

    seen = set()

    for rank, document in enumerate(
        documents,
        start=1,
    ):

        doc_id = _get_doc_id(document)

        if doc_id is None:
            continue

        # Prevent duplicate document IDs from corrupting
        # rank-based fusion.
        if doc_id in seen:
            continue

        seen.add(doc_id)

        ranking.append(
            {
                "doc_id": doc_id,
                "rank": rank,
                "score": _get_score(document),
                "document": document,
            }
        )

    return ranking


# ==========================================================
# MIN-MAX FUSION
# ==========================================================

def minmax_fusion(
    dense_documents: List[Dict[str, Any]],
    bm25_documents: List[Dict[str, Any]],
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
) -> List[Dict[str, Any]]:
    """
    Perform weighted Min-Max fusion.

    Dense and BM25 scores are normalized independently
    for the current query.

    This is the CURRENT BASELINE being evaluated.

    It intentionally preserves the query-local normalization
    behaviour so that Optimization #6 measures whether RRF
    is more robust.

    Documents missing from one retriever receive score 0
    for that retriever.
    """

    if dense_weight < 0 or bm25_weight < 0:
        raise ValueError(
            "Fusion weights must be non-negative."
        )

    if dense_weight + bm25_weight == 0:
        raise ValueError(
            "At least one fusion weight must be > 0."
        )

    dense = prepare_ranking(
        dense_documents
    )

    bm25 = prepare_ranking(
        bm25_documents
    )

    dense_scores = _normalize_scores(
        [
            item["score"]
            for item in dense
        ]
    )

    bm25_scores = _normalize_scores(
        [
            item["score"]
            for item in bm25
        ]
    )

    dense_map = {}
    bm25_map = {}

    document_map = {}

    for item, normalized_score in zip(
        dense,
        dense_scores,
    ):

        doc_id = item["doc_id"]

        dense_map[doc_id] = normalized_score
        document_map[doc_id] = item["document"]

    for item, normalized_score in zip(
        bm25,
        bm25_scores,
    ):

        doc_id = item["doc_id"]

        bm25_map[doc_id] = normalized_score

        if doc_id not in document_map:
            document_map[doc_id] = item["document"]

    all_doc_ids = (
        set(dense_map.keys())
        |
        set(bm25_map.keys())
    )

    fused = []

    for doc_id in all_doc_ids:

        dense_score = dense_map.get(
            doc_id,
            0.0,
        )

        sparse_score = bm25_map.get(
            doc_id,
            0.0,
        )

        fusion_score = (
            dense_weight * dense_score
            +
            bm25_weight * sparse_score
        )

        fused.append(
            {
                "doc_id": doc_id,
                "document": document_map[doc_id],
                "dense_score": dense_score,
                "bm25_score": sparse_score,
                "fusion_score": fusion_score,
            }
        )

    # Stable deterministic ordering:
    #
    # 1. fusion score
    # 2. dense score
    # 3. BM25 score
    # 4. doc ID
    #
    fused.sort(
        key=lambda item: (
            item["fusion_score"],
            item["dense_score"],
            item["bm25_score"],
            -int(item["doc_id"])
            if str(item["doc_id"]).isdigit()
            else 0,
        ),
        reverse=True,
    )

    for rank, item in enumerate(
        fused,
        start=1,
    ):
        item["rank"] = rank

    return fused


# ==========================================================
# RRF
# ==========================================================

def rrf_fusion(
    dense_documents: List[Dict[str, Any]],
    bm25_documents: List[Dict[str, Any]],
    rrf_k: int = DEFAULT_RRF_K,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion.

    RRF score:

        RRF(d) =
            1 / (rrf_k + rank_dense)
            +
            1 / (rrf_k + rank_bm25)

    RRF uses ranks rather than raw scores.

    Therefore it avoids the score-scale mismatch between
    dense similarity and BM25 scores.
    """

    if rrf_k <= 0:
        raise ValueError(
            "rrf_k must be greater than 0."
        )

    dense = prepare_ranking(
        dense_documents
    )

    bm25 = prepare_ranking(
        bm25_documents
    )

    scores: Dict[Any, float] = {}
    document_map: Dict[Any, Dict[str, Any]] = {}

    dense_rank_map = {}
    bm25_rank_map = {}

    for item in dense:

        doc_id = item["doc_id"]

        dense_rank_map[doc_id] = item["rank"]
        document_map[doc_id] = item["document"]

        scores[doc_id] = (
            scores.get(doc_id, 0.0)
            +
            1.0 / (
                rrf_k + item["rank"]
            )
        )

    for item in bm25:

        doc_id = item["doc_id"]

        bm25_rank_map[doc_id] = item["rank"]

        if doc_id not in document_map:
            document_map[doc_id] = item["document"]

        scores[doc_id] = (
            scores.get(doc_id, 0.0)
            +
            1.0 / (
                rrf_k + item["rank"]
            )
        )

    fused = []

    for doc_id, score in scores.items():

        fused.append(
            {
                "doc_id": doc_id,
                "document": document_map[doc_id],
                "dense_rank": dense_rank_map.get(
                    doc_id
                ),
                "bm25_rank": bm25_rank_map.get(
                    doc_id
                ),
                "fusion_score": score,
            }
        )

    fused.sort(
        key=lambda item: (
            item["fusion_score"],
            -(
                item["dense_rank"]
                if item["dense_rank"] is not None
                else 10**9
            ),
            -(
                item["bm25_rank"]
                if item["bm25_rank"] is not None
                else 10**9
            ),
        ),
        reverse=True,
    )

    for rank, item in enumerate(
        fused,
        start=1,
    ):
        item["rank"] = rank

    return fused


# ==========================================================
# GENERAL FUSION ENTRY POINT
# ==========================================================

def fuse(
    dense_documents: List[Dict[str, Any]],
    bm25_documents: List[Dict[str, Any]],
    method: str = "minmax",
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    rrf_k: int = DEFAULT_RRF_K,
) -> List[Dict[str, Any]]:
    """
    Unified fusion interface.

    method:

        "minmax"
        "rrf"
    """

    method = method.lower().strip()

    if method == "minmax":

        return minmax_fusion(
            dense_documents,
            bm25_documents,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )

    if method == "rrf":

        return rrf_fusion(
            dense_documents,
            bm25_documents,
            rrf_k=rrf_k,
        )

    raise ValueError(
        f"Unsupported fusion method: {method}. "
        f"Expected 'minmax' or 'rrf'."
    )