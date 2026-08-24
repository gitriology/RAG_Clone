# ==========================================================
# RECIPROCAL RANK FUSION (RRF)
# Optimization #6
# ==========================================================

from typing import Dict, List, Any


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60,
) -> Dict[Any, float]:

    """
    Reciprocal Rank Fusion.

    RRF does not directly combine dense and sparse scores.

    Instead, it combines the rankings produced by the
    dense retriever and sparse retriever.

    Formula:

        RRF(d) =
            1 / (k + rank_dense(d))
            +
            1 / (k + rank_sparse(d))

    Parameters
    ----------
    dense_results:
        Ordered dense retrieval results.

    sparse_results:
        Ordered sparse/BM25 retrieval results.

    k:
        RRF rank constant.
        Standard value is 60.

    Returns
    -------
    Dict mapping document ID -> RRF score.
    """

    scores = {}

    # ------------------------------------------------------
    # Dense ranking
    # ------------------------------------------------------

    for rank, result in enumerate(
        dense_results,
        start=1
    ):

        doc_id = result["doc_id"]

        scores.setdefault(
            doc_id,
            0.0
        )

        scores[doc_id] += (
            1.0 /
            (k + rank)
        )

    # ------------------------------------------------------
    # Sparse ranking
    # ------------------------------------------------------

    for rank, result in enumerate(
        sparse_results,
        start=1
    ):

        doc_id = result["doc_id"]

        scores.setdefault(
            doc_id,
            0.0
        )

        scores[doc_id] += (
            1.0 /
            (k + rank)
        )

    return scores


def rank_rrf_results(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:

    """
    Produce a ranked list using RRF.

    The original dense/sparse scores are preserved
    so downstream MS-ARC and debugging can still
    inspect them.
    """

    rrf_scores = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        k=k,
    )

    # ------------------------------------------------------
    # Create lookup tables
    # ------------------------------------------------------

    dense_lookup = {
        item["doc_id"]: item
        for item in dense_results
    }

    sparse_lookup = {
        item["doc_id"]: item
        for item in sparse_results
    }

    all_doc_ids = set(
        dense_lookup.keys()
    ) | set(
        sparse_lookup.keys()
    )

    ranked = []

    # ------------------------------------------------------
    # Build merged result
    # ------------------------------------------------------

    for doc_id in all_doc_ids:

        dense_item = dense_lookup.get(
            doc_id
        )

        sparse_item = sparse_lookup.get(
            doc_id
        )

        dense_score = (
            dense_item.get(
                "score",
                0.0
            )
            if dense_item
            else 0.0
        )

        sparse_score = (
            sparse_item.get(
                "score",
                0.0
            )
            if sparse_item
            else 0.0
        )

        ranked.append({

            "doc_id":
                doc_id,

            "rrf_score":
                rrf_scores[
                    doc_id
                ],

            "dense_score":
                float(
                    dense_score
                ),

            "sparse_score":
                float(
                    sparse_score
                ),

        })

    # ------------------------------------------------------
    # Sort by RRF score
    # ------------------------------------------------------

    ranked.sort(

        key=lambda x:
            x["rrf_score"],

        reverse=True

    )

    return ranked