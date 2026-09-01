"""
Optimization #6
Hybrid Search / Fusion Layer

Supports:

    1. Min-Max weighted fusion
    2. Reciprocal Rank Fusion (RRF)

Important:
    This module is responsible ONLY for candidate retrieval
    and fusion.

MS-ARC, stability analysis and final reranking are handled
outside this module.
"""

import numpy as np
import re

from backend.retrieval.dense.embedder import encode_query
from backend.retrieval.query_matching import (
    analyze_query,
    lexical_match_score,
    reference_present,
)


# ==========================================================
# NORMALIZATION
# ==========================================================

def _normalize(scores):
    """
    Query-local Min-Max normalization.

    NOTE:
        This is intentionally retained for the Min-Max
        ablation experiment.

        It should NOT be interpreted as globally calibrated
        scoring.
    """

    scores = np.asarray(
        scores,
        dtype=np.float32,
    )

    if len(scores) == 0:
        return scores

    minimum = np.min(scores)
    maximum = np.max(scores)

    if maximum == minimum:
        return np.ones_like(scores)

    return (
        (scores - minimum)
        / (maximum - minimum)
    )


# ==========================================================
# RRF
# ==========================================================

def _rrf_score(
    dense_rank,
    sparse_rank,
    rrf_k=60,
):
    """
    Reciprocal Rank Fusion.

        RRF(d) =
            1 / (k + dense_rank)
            +
            1 / (k + sparse_rank)

    A document contributes only from the retrieval
    systems in which it appears.
    """

    score = 0.0

    if dense_rank is not None:

        score += (
            1.0
            / (rrf_k + dense_rank)
        )

    if sparse_rank is not None:

        score += (
            1.0
            / (rrf_k + sparse_rank)
        )

    return score


# ==========================================================
# LEXICAL ANCHOR RECOVERY
# ==========================================================


def _anchor_candidates(query, texts, limit_per_focus=4):
    """Recover exact named/reference matches that min-max fusion can hide.

    This is intentionally independent of dense similarity. It is especially
    useful for legal references (Article 12), acronyms and multi-word names.
    """
    analysis = analyze_query(query)
    focus = analysis.get("focus_phrases", [])
    references = set(analysis.get("references", []))
    selected = {}

    for phrase in focus:
        scored = []
        for idx, text in enumerate(texts):
            if phrase in references:
                ok = reference_present(text, phrase)
            else:
                detail = lexical_match_score(phrase, text)
                ok = phrase in detail.get("matched", [])
            if not ok:
                continue

            # Prefer a real provision/definition over a table-of-contents-only
            # hit when the reference is legal/numbered.
            score = 1.0
            lower = str(text).lower()
            if phrase in references and re.search(rf"(?<!\\d){re.escape(phrase.split()[-1])}\\s*[.)]\\s*[a-z]", lower):
                score += 0.20
            if " definition" in lower or " means " in lower or " refers to " in lower:
                score += 0.10
            scored.append((score, idx))

        scored.sort(reverse=True)
        for _, idx in scored[:limit_per_focus]:
            selected[idx] = True

    return list(selected.keys())


# ==========================================================
# HYBRID SEARCH
# ==========================================================

def hybrid_search(
    query,
    faiss_index,
    bm25,
    texts,
    k=10,
    candidate_k=None,
    dense_weight=0.6,
    sparse_weight=0.4,
    fusion_method="minmax",
    rrf_k=60,
):
    """
    Execute dense + sparse retrieval followed by fusion.

    Supported:

        fusion_method="minmax"
        fusion_method="rrf"

    Returns:

        {
            "dense_results": [...],
            "sparse_results": [...],
            "merged_results": [...],
            "fusion_method": ...,
            "candidate_k": ...
        }
    """

    # ======================================================
    # NORMALIZE FUSION METHOD
    # ======================================================

    fusion_method = (
        fusion_method
        or "minmax"
    ).lower().strip()

    if fusion_method not in {
        "minmax",
        "rrf",
    }:

        raise ValueError(
            f"Unsupported fusion method: "
            f"{fusion_method}. "
            f"Expected 'minmax' or 'rrf'."
        )

    # ======================================================
    # CANDIDATE DEPTH
    # ======================================================

    if candidate_k is None:

        candidate_k = max(
            k * 4,
            20,
        )

    candidate_k = max(
        int(candidate_k),
        int(k),
    )

    print(
        f"[Hybrid Retrieval] "
        f"Executing hybrid search: "
        f"k={k}, "
        f"candidate_k={candidate_k}, "
        f"fusion={fusion_method}"
    )

    # ======================================================
    # DENSE RETRIEVAL
    # ======================================================

    query_embedding = encode_query(query)

    dense_scores, dense_ids = (
        faiss_index.search(
            query_embedding,
            candidate_k,
        )
    )

    dense_scores_raw = dense_scores[0]
    dense_ids = dense_ids[0]

    valid_dense = []

    for doc_id, score in zip(
        dense_ids,
        dense_scores_raw,
    ):

        doc_id = int(doc_id)

        if doc_id < 0:
            continue

        if doc_id >= len(texts):
            continue

        valid_dense.append(
            (
                doc_id,
                float(score),
            )
        )

    # ======================================================
    # DENSE NORMALIZATION
    # ======================================================

    normalized_dense = _normalize(
        [
            score
            for _, score in valid_dense
        ]
    )

    dense_results = []
    dense_lookup = {}
    dense_rank_lookup = {}

    for rank, (
        (doc_id, raw_score),
        normalized_score,
    ) in enumerate(
        zip(
            valid_dense,
            normalized_dense,
        ),
        start=1,
    ):

        dense_rank_lookup[doc_id] = rank

        document = {

            "doc_id": doc_id,

            "text": texts[doc_id],

            "dense_raw_score":
                raw_score,

            "dense_score":
                float(normalized_score),

            "bm25_raw_score":
                0.0,

            "bm25_score":
                0.0,

            "hybrid_score":
                0.0,

            "dense_rank":
                rank,

            "sparse_rank":
                None,

            "fusion_method":
                fusion_method,

        }

        dense_results.append(document)

        dense_lookup[doc_id] = document

    # ======================================================
    # SPARSE RETRIEVAL
    # ======================================================

    bm25_tokens = query.lower().split()

    print(
        f"[Hybrid Retrieval] "
        f"BM25 tokens: {bm25_tokens}"
    )

    bm25_scores = bm25.get_scores(
        bm25_tokens
    )

    bm25_ids = np.argsort(
        bm25_scores
    )[::-1][:candidate_k]

    valid_sparse = []

    for doc_id in bm25_ids:

        doc_id = int(doc_id)

        if doc_id < 0:
            continue

        if doc_id >= len(texts):
            continue

        valid_sparse.append(
            (
                doc_id,
                float(
                    bm25_scores[doc_id]
                ),
            )
        )

    normalized_sparse = _normalize(
        [
            score
            for _, score in valid_sparse
        ]
    )

    sparse_results = []
    sparse_lookup = {}
    sparse_rank_lookup = {}

    for rank, (
        (doc_id, raw_score),
        normalized_score,
    ) in enumerate(
        zip(
            valid_sparse,
            normalized_sparse,
        ),
        start=1,
    ):

        sparse_rank_lookup[doc_id] = rank

        document = {

            "doc_id": doc_id,

            "text": texts[doc_id],

            "dense_raw_score":
                0.0,

            "dense_score":
                0.0,

            "bm25_raw_score":
                raw_score,

            "bm25_score":
                float(normalized_score),

            "hybrid_score":
                0.0,

            "dense_rank":
                None,

            "sparse_rank":
                rank,

            "fusion_method":
                fusion_method,

        }

        sparse_results.append(document)

        sparse_lookup[doc_id] = document

    # ======================================================
    # MERGE CANDIDATES
    # ======================================================

    all_doc_ids = set(
        dense_lookup.keys()
    )

    all_doc_ids.update(
        sparse_lookup.keys()
    )

    print(
        f"[Hybrid Retrieval] "
        f"Dense candidates: "
        f"{len(dense_results)}"
    )

    print(
        f"[Hybrid Retrieval] "
        f"Sparse candidates: "
        f"{len(sparse_results)}"
    )

    print(
        f"[Hybrid Retrieval] "
        f"Merged candidates: "
        f"{len(all_doc_ids)}"
    )

    # Exact lexical/reference anchors are added to the candidate pool before
    # fusion. This prevents an exact answer from being lost just because a
    # long chunk received a slightly lower dense/min-max score.
    anchor_ids = _anchor_candidates(
        query,
        texts,
        limit_per_focus=4,
    )
    all_doc_ids.update(anchor_ids)

    if anchor_ids:
        print(
            f"[Hybrid Retrieval] Lexical anchor candidates: {len(anchor_ids)}"
        )

    # ======================================================
    # FUSION
    # ======================================================

    merged_results = []

    for doc_id in all_doc_ids:

        dense_doc = dense_lookup.get(
            doc_id
        )

        sparse_doc = sparse_lookup.get(
            doc_id
        )

        dense_score = (
            dense_doc["dense_score"]
            if dense_doc
            else 0.0
        )

        sparse_score = (
            sparse_doc["bm25_score"]
            if sparse_doc
            else 0.0
        )

        dense_raw_score = (
            dense_doc["dense_raw_score"]
            if dense_doc
            else 0.0
        )

        sparse_raw_score = (
            sparse_doc["bm25_raw_score"]
            if sparse_doc
            else 0.0
        )

        dense_rank = (
            dense_rank_lookup.get(
                doc_id
            )
        )

        sparse_rank = (
            sparse_rank_lookup.get(
                doc_id
            )
        )

        lexical_detail = lexical_match_score(
            query,
            texts[doc_id],
        )
        lexical_anchor_score = float(
            lexical_detail.get("score", 0.0)
        )
        reference_match = bool(
            lexical_detail.get("reference_match", False)
        )
        exact_phrase_match = bool(
            lexical_detail.get("exact_phrase", False)
        )

        # --------------------------------------------------
        # MIN-MAX FUSION
        # --------------------------------------------------

        if fusion_method == "minmax":

            hybrid_score = (

                dense_weight
                * dense_score

                +

                sparse_weight
                * sparse_score

            )

        # --------------------------------------------------
        # RRF
        # --------------------------------------------------

        else:

            hybrid_score = _rrf_score(
                dense_rank=dense_rank,
                sparse_rank=sparse_rank,
                rrf_k=rrf_k,
            )

        # Keep the existing fusion score intact as a diagnostic, then use a
        # small query-grounding boost for ranking. Exact references are the
        # strongest anchor; ordinary exact phrases are next.
        ranking_score = float(hybrid_score)
        if lexical_anchor_score > 0.0:
            ranking_score += 0.18 * lexical_anchor_score
        if exact_phrase_match:
            ranking_score += 0.10
        if reference_match:
            ranking_score += 0.30

        merged_results.append({

            "doc_id":
                int(doc_id),

            "text":
                texts[doc_id],

            "dense_raw_score":
                float(dense_raw_score),

            "dense_score":
                float(dense_score),

            "bm25_raw_score":
                float(sparse_raw_score),

            "bm25_score":
                float(sparse_score),

            "hybrid_score":
                float(hybrid_score),

            "ranking_score":
                float(ranking_score),

            "lexical_anchor_score":
                lexical_anchor_score,

            "exact_phrase_match":
                exact_phrase_match,

            "reference_match":
                reference_match,

            "matched_focus":
                list(lexical_detail.get("matched", [])),

            "dense_rank":
                dense_rank,

            "sparse_rank":
                sparse_rank,

            "fusion_method":
                fusion_method,

        })

    # ======================================================
    # SORT
    # ======================================================

    merged_results.sort(
        key=lambda x: (
            x.get("reference_match", False),
            x.get("exact_phrase_match", False),
            x.get("lexical_anchor_score", 0.0),
            x.get("ranking_score", x.get("hybrid_score", 0.0)),
            x.get("hybrid_score", 0.0),
        ),
        reverse=True,
    )

    # ======================================================
    # TOP K
    # ======================================================

    merged_results = (
        merged_results[:k]
    )

    print(
        f"[Hybrid Retrieval] "
        f"Final documents: "
        f"{len(merged_results)}"
    )

    # ======================================================
    # DEBUG
    # ======================================================

    for rank, document in enumerate(
        merged_results,
        start=1,
    ):

        print(
            f"[Hybrid Retrieval] "
            f"Top-{rank} "
            f"doc_id={document['doc_id']} "
            f"dense={document['dense_score']:.4f} "
            f"bm25={document['bm25_score']:.4f} "
            f"hybrid={document['hybrid_score']:.6f} "
            f"anchor={document.get('lexical_anchor_score', 0.0):.4f} "
            f"dense_rank={document['dense_rank']} "
            f"sparse_rank={document['sparse_rank']}"
        )

    # ======================================================
    # RETURN
    # ======================================================

    return {

        "dense_results":
            dense_results,

        "sparse_results":
            sparse_results,

        "merged_results":
            merged_results,

        "fusion_method":
            fusion_method,

        "candidate_k":
            candidate_k,

    }