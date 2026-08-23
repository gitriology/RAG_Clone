import re

import numpy as np

from backend.retrieval.dense.embedder import encode_query


# ==========================================================
# QUERY NORMALIZATION
# ==========================================================

QUESTION_WORDS = {

    "what",
    "is",
    "are",
    "was",
    "were",
    "who",
    "whom",
    "where",
    "when",
    "why",
    "how",
    "can",
    "could",
    "would",
    "should",
    "tell",
    "explain",
    "define",
    "give",
    "me",
    "about",

}


def _normalize_query(query):

    """
    Normalize the query for sparse/BM25 retrieval.

    Dense retrieval receives the ORIGINAL query.
    """

    if not query:
        return ""

    query = str(
        query
    ).lower().strip()

    query = re.sub(
        r"[^a-z0-9\s\-]",
        " ",
        query,
    )

    tokens = query.split()

    filtered_tokens = [

        token

        for token in tokens

        if token not in QUESTION_WORDS

    ]

    if not filtered_tokens:

        filtered_tokens = tokens

    return " ".join(
        filtered_tokens
    )


# ==========================================================
# MIN-MAX NORMALIZATION
# ==========================================================

def _normalize(scores):

    """
    Query-local Min-Max normalization.

    This is intentionally retained as the BASELINE
    fusion strategy for Optimization #6.

    It should NOT be silently replaced because it is
    required for the fusion ablation experiment.
    """

    scores = np.asarray(
        scores,
        dtype=np.float32,
    )

    if len(scores) == 0:

        return scores

    minimum = np.min(
        scores
    )

    maximum = np.max(
        scores
    )

    if maximum == minimum:

        return np.ones_like(
            scores
        )

    return (
        scores - minimum
    ) / (
        maximum - minimum
    )


# ==========================================================
# Z-SCORE NORMALIZATION
# ==========================================================

def _zscore_normalize(scores):

    """
    Standard-score normalization.

    This is provided as an optional calibrated-style
    score fusion baseline.

    Unlike Min-Max, it preserves information about how
    far a score lies from the distribution mean.
    """

    scores = np.asarray(
        scores,
        dtype=np.float32,
    )

    if len(scores) == 0:

        return scores

    mean = np.mean(
        scores
    )

    std = np.std(
        scores
    )

    if std < 1e-8:

        return np.zeros_like(
            scores
        )

    z = (
        scores - mean
    ) / std

    # ------------------------------------------------------
    # Convert z-score to a bounded [0,1] score.
    #
    # Logistic transformation prevents large outliers
    # from dominating fusion.
    # ------------------------------------------------------

    return 1.0 / (
        1.0 + np.exp(
            -z
        )
    )


# ==========================================================
# RRF SCORE
# ==========================================================

def _rrf_score(
    dense_rank,
    sparse_rank,
    rrf_k=60,
):

    """
    Reciprocal Rank Fusion.

    Formula:

        1 / (rrf_k + dense_rank)
        +
        1 / (rrf_k + sparse_rank)

    Ranks are 1-based.

    Missing rankings contribute zero.
    """

    score = 0.0

    if dense_rank is not None:

        score += 1.0 / (
            rrf_k + dense_rank
        )

    if sparse_rank is not None:

        score += 1.0 / (
            rrf_k + sparse_rank
        )

    return score


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
    Hybrid Retrieval.

    Fusion methods:

        minmax
            Original weighted Min-Max fusion.
            This is the baseline.

        rrf
            Reciprocal Rank Fusion.

        zscore
            Z-score + sigmoid score fusion.

    Optimization #6
    ----------------
    candidate_k is supplied by MS-ARC.

    The old:

        max(k * 4, 20)

    rule is retained ONLY as a fallback for direct calls
    that bypass MS-ARC. The MS-ARC pipeline always supplies
    an adaptive candidate_k.
    """

    # ======================================================
    # ORIGINAL QUERY
    # ======================================================

    original_query = str(
        query
    ).strip()

    # ======================================================
    # SPARSE QUERY
    # ======================================================

    sparse_query = _normalize_query(
        original_query
    )

    print(
        "[Hybrid Retrieval] "
        f"Original query: {original_query}"
    )

    print(
        "[Hybrid Retrieval] "
        f"Sparse query: {sparse_query}"
    )

    # ======================================================
    # Candidate Pool
    # ======================================================

    if candidate_k is None:

        # --------------------------------------------------
        # Direct-call fallback only.
        #
        # MS-ARC itself never uses this path.
        # --------------------------------------------------

        candidate_k = max(
            k * 4,
            20,
        )

        print(
            "[Hybrid Retrieval] "
            "WARNING: candidate_k not supplied. "
            "Using direct-call baseline."
        )

    candidate_k = max(
        int(candidate_k),
        k,
    )

    candidate_k = min(
        candidate_k,
        len(texts),
    )

    print(
        f"[Hybrid Retrieval] "
        f"Executing hybrid search "
        f"with k={k}, "
        f"candidate_k={candidate_k}, "
        f"fusion={fusion_method}"
    )

    # ======================================================
    # Dense Retrieval
    # ======================================================

    query_embedding = encode_query(
        original_query
    )

    dense_raw_scores, dense_ids = (
        faiss_index.search(
            query_embedding,
            candidate_k,
        )
    )

    dense_raw_scores = (
        dense_raw_scores[0]
    )

    dense_ids = (
        dense_ids[0]
    )

    # ------------------------------------------------------
    # Keep raw scores.
    #
    # They are needed for:
    #
    #   1. Min-Max fusion
    #   2. Z-score fusion
    #   3. diagnostics
    # ------------------------------------------------------

    dense_normalized_scores = _normalize(
        dense_raw_scores
    )

    dense_zscores = _zscore_normalize(
        dense_raw_scores
    )

    dense_results = []

    dense_lookup = {}

    dense_rank_lookup = {}

    for rank, (
        doc_id,
        normalized_score,
        raw_score,
        zscore,
    ) in enumerate(

        zip(
            dense_ids,
            dense_normalized_scores,
            dense_raw_scores,
            dense_zscores,
        ),

        start=1,
    ):

        doc_id = int(
            doc_id
        )

        if (
            doc_id < 0
            or doc_id >= len(texts)
        ):

            continue

        doc = {

            "doc_id":
                doc_id,

            "text":
                texts[doc_id],

            "dense_score":
                float(normalized_score),

            "dense_raw_score":
                float(raw_score),

            "dense_zscore":
                float(zscore),

            "bm25_score":
                0.0,

            "bm25_raw_score":
                0.0,

            "bm25_zscore":
                0.0,

            "hybrid_score":
                0.0,

        }

        dense_results.append(
            doc
        )

        dense_lookup[
            doc_id
        ] = doc

        dense_rank_lookup[
            doc_id
        ] = rank

    # ======================================================
    # Sparse Retrieval
    # ======================================================

    sparse_tokens = (

        sparse_query.split()

        if sparse_query

        else original_query.lower().split()

    )

    print(
        "[Hybrid Retrieval] "
        f"BM25 tokens: {sparse_tokens}"
    )

    bm25_scores = bm25.get_scores(
        sparse_tokens
    )

    bm25_ids = np.argsort(
        bm25_scores
    )[::-1][:candidate_k]

    bm25_raw_top_scores = (
        bm25_scores[
            bm25_ids
        ]
    )

    bm25_normalized_scores = _normalize(
        bm25_raw_top_scores
    )

    bm25_zscores = _zscore_normalize(
        bm25_raw_top_scores
    )

    sparse_results = []

    sparse_lookup = {}

    sparse_rank_lookup = {}

    for rank, (
        doc_id,
        normalized_score,
        raw_score,
        zscore,
    ) in enumerate(

        zip(
            bm25_ids,
            bm25_normalized_scores,
            bm25_raw_top_scores,
            bm25_zscores,
        ),

        start=1,
    ):

        doc_id = int(
            doc_id
        )

        if (
            doc_id < 0
            or doc_id >= len(texts)
        ):

            continue

        doc = {

            "doc_id":
                doc_id,

            "text":
                texts[doc_id],

            "dense_score":
                0.0,

            "dense_raw_score":
                0.0,

            "dense_zscore":
                0.0,

            "bm25_score":
                float(normalized_score),

            "bm25_raw_score":
                float(raw_score),

            "bm25_zscore":
                float(zscore),

            "hybrid_score":
                0.0,

        }

        sparse_results.append(
            doc
        )

        sparse_lookup[
            doc_id
        ] = doc

        sparse_rank_lookup[
            doc_id
        ] = rank

    # ======================================================
    # MERGE CANDIDATES
    # ======================================================

    merged_lookup = {}

    all_doc_ids = set(
        dense_lookup.keys()
    )

    all_doc_ids.update(
        sparse_lookup.keys()
    )

    # ======================================================
    # HYBRID FUSION
    # ======================================================

    fusion_method = str(
        fusion_method
    ).lower().strip()

    for doc_id in all_doc_ids:

        dense_doc = dense_lookup.get(
            doc_id
        )

        sparse_doc = sparse_lookup.get(
            doc_id
        )

        # --------------------------------------------------
        # Dense scores
        # --------------------------------------------------

        dense_score = (

            dense_doc.get(
                "dense_score",
                0.0,
            )

            if dense_doc

            else 0.0

        )

        dense_raw_score = (

            dense_doc.get(
                "dense_raw_score",
                0.0,
            )

            if dense_doc

            else 0.0

        )

        dense_zscore = (

            dense_doc.get(
                "dense_zscore",
                0.0,
            )

            if dense_doc

            else 0.0

        )

        # --------------------------------------------------
        # Sparse scores
        # --------------------------------------------------

        sparse_score = (

            sparse_doc.get(
                "bm25_score",
                0.0,
            )

            if sparse_doc

            else 0.0

        )

        sparse_raw_score = (

            sparse_doc.get(
                "bm25_raw_score",
                0.0,
            )

            if sparse_doc

            else 0.0

        )

        sparse_zscore = (

            sparse_doc.get(
                "bm25_zscore",
                0.0,
            )

            if sparse_doc

            else 0.0

        )

        # ==================================================
        # MIN-MAX FUSION
        # ==================================================

        if fusion_method == "minmax":

            hybrid_score = (

                dense_weight
                * dense_score

                +

                sparse_weight
                * sparse_score

            )

        # ==================================================
        # RRF
        # ==================================================

        elif fusion_method == "rrf":

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

            hybrid_score = _rrf_score(

                dense_rank=dense_rank,

                sparse_rank=sparse_rank,

                rrf_k=rrf_k,

            )

        # ==================================================
        # Z-SCORE FUSION
        # ==================================================

        elif fusion_method in (
            "zscore",
            "calibrated",
        ):

            hybrid_score = (

                dense_weight
                * dense_zscore

                +

                sparse_weight
                * sparse_zscore

            )

        else:

            raise ValueError(

                "Unknown fusion_method: "

                f"{fusion_method}. "

                "Supported values are: "

                "'minmax', 'rrf', 'zscore'."

            )

        # ==================================================
        # Store
        # ==================================================

        merged_lookup[
            doc_id
        ] = {

            "doc_id":
                doc_id,

            "text":
                texts[doc_id],

            "dense_score":
                float(dense_score),

            "dense_raw_score":
                float(dense_raw_score),

            "dense_zscore":
                float(dense_zscore),

            "bm25_score":
                float(sparse_score),

            "bm25_raw_score":
                float(sparse_raw_score),

            "bm25_zscore":
                float(sparse_zscore),

            "hybrid_score":
                float(hybrid_score),

            "dense_rank":
                dense_rank_lookup.get(
                    doc_id
                ),

            "sparse_rank":
                sparse_rank_lookup.get(
                    doc_id
                ),

        }

    # ======================================================
    # SORT
    # ======================================================

    merged_results = sorted(

        merged_lookup.values(),

        key=lambda x:
            x["hybrid_score"],

        reverse=True,

    )

    # ======================================================
    # FINAL TOP-K
    # ======================================================

    final_results = (
        merged_results[:k]
    )

    # ======================================================
    # DEBUG
    # ======================================================

    print(
        "[Hybrid Retrieval] "
        f"Dense candidates: "
        f"{len(dense_results)}"
    )

    print(
        "[Hybrid Retrieval] "
        f"Sparse candidates: "
        f"{len(sparse_results)}"
    )

    print(
        "[Hybrid Retrieval] "
        f"Merged candidates: "
        f"{len(merged_lookup)}"
    )

    print(
        "[Hybrid Retrieval] "
        f"Final documents: "
        f"{len(final_results)}"
    )

    print(
        "[Hybrid Retrieval] "
        f"Fusion method: "
        f"{fusion_method}"
    )

    # ======================================================
    # TOP DOCUMENT DIAGNOSTICS
    # ======================================================

    for rank, doc in enumerate(
        final_results,
        start=1,
    ):

        print(

            f"[Hybrid Retrieval] "

            f"Top-{rank} "

            f"doc_id={doc['doc_id']} "

            f"dense={doc['dense_score']:.4f} "

            f"bm25={doc['bm25_score']:.4f} "

            f"hybrid={doc['hybrid_score']:.4f}"

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
            final_results,

        "candidate_k":
            candidate_k,

        "fusion_method":
            fusion_method,

        "dense_weight":
            dense_weight,

        "sparse_weight":
            sparse_weight,

        "rrf_k":
            rrf_k,

    }