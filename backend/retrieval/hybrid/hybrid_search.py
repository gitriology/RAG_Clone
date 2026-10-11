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

import copy
import hashlib
from collections import OrderedDict

import numpy as np
import re
from time import perf_counter

from backend.retrieval.dense.embedder import encode_query
from backend.retrieval.lexical.bm25 import tokenize
from backend.retrieval.query_matching import (
    analyze_query,
    lexical_match_score,
    reference_present,
)


# ==========================================================
# QUERY RESULT CACHE
# ==========================================================

_CACHE_MAX_SIZE = 128
_QUERY_CACHE = OrderedDict()
_CORPUS_FINGERPRINT_CACHE = {}


def _corpus_fingerprint(texts):
    """Fingerprint a corpus once, recomputing only if its string objects change.

    The corpus is immutable during normal serving. Comparing string identities
    is much cheaper than hashing every full chunk for every retrieval iteration,
    while still invalidating the cache if a list entry is replaced or resized.
    """
    corpus_id = id(texts)
    signature = (len(texts), tuple(id(text) for text in texts))
    cached = _CORPUS_FINGERPRINT_CACHE.get(corpus_id)
    if cached is not None and cached[0] == signature:
        return cached[1]

    digest = hashlib.sha256()
    digest.update(str(len(texts)).encode("utf-8"))
    for text in texts:
        encoded = str(text).encode("utf-8", errors="replace")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    fingerprint = digest.hexdigest()
    _CORPUS_FINGERPRINT_CACHE[corpus_id] = (signature, fingerprint)

    # Bound the auxiliary cache in long-running processes.
    while len(_CORPUS_FINGERPRINT_CACHE) > 16:
        _CORPUS_FINGERPRINT_CACHE.pop(next(iter(_CORPUS_FINGERPRINT_CACHE)))
    return fingerprint


_ANCHOR_CACHE_MAX_SIZE = 256
_ANCHOR_CACHE = OrderedDict()

# Query embeddings are independent of K/candidate depth, so adaptive retrieval
# can reuse them safely across iterations. Keep this cache bounded and return
# copies so callers cannot mutate cached arrays.
_QUERY_EMBEDDING_CACHE_MAX_SIZE = 512
_QUERY_EMBEDDING_CACHE = OrderedDict()
_QUERY_EMBEDDING_STATS = {"hits": 0, "misses": 0, "evictions": 0}


def _embedding_model_cache_identity():
    """Stable cache namespace for configured model version and encode settings.

    Set RAGAR_EMBEDDING_MODEL_REVISION when the deployed model/revision changes.
    A stable namespace is intentional: deriving the key from whether a lazy model
    has loaded would cause an avoidable miss on the first warm request.
    """
    import os
    model_name = "BAAI/bge-base-en-v1.5"
    model_revision = os.getenv("RAGAR_EMBEDDING_MODEL_REVISION", "configured-default")
    # Query encoding currently uses normalized float32 vectors. Include these
    # settings so future configuration changes cannot silently reuse old keys.
    return (model_name, model_revision, "normalize=true", "dtype=float32", "encode_query-v1")


def _get_query_embedding(query):
    normalized_query = str(query).strip()
    key = (_embedding_model_cache_identity(), normalized_query)
    cached = _QUERY_EMBEDDING_CACHE.get(key)
    if cached is not None:
        _QUERY_EMBEDDING_CACHE.move_to_end(key)
        _QUERY_EMBEDDING_STATS["hits"] += 1
        return cached.copy(), "hit"
    _QUERY_EMBEDDING_STATS["misses"] += 1
    embedding = np.asarray(encode_query(normalized_query), dtype=np.float32)
    _QUERY_EMBEDDING_CACHE[key] = embedding.copy()
    _QUERY_EMBEDDING_CACHE.move_to_end(key)
    while len(_QUERY_EMBEDDING_CACHE) > _QUERY_EMBEDDING_CACHE_MAX_SIZE:
        _QUERY_EMBEDDING_CACHE.popitem(last=False)
        _QUERY_EMBEDDING_STATS["evictions"] += 1
    return embedding, "miss"


def _cache_key(query, faiss_index, bm25, texts, k, candidate_k, dense_weight, sparse_weight, fusion_method, rrf_k, return_candidates=False):
    """Create a cache key that invalidates when corpus or output mode changes."""
    return (
        str(query), id(faiss_index), id(bm25), _corpus_fingerprint(texts),
        int(k), int(candidate_k), float(dense_weight), float(sparse_weight),
        str(fusion_method), int(rrf_k), bool(return_candidates),
    )


def clear_result_cache():
    """Clear result/anchor caches but retain query embeddings for warm tests."""
    _QUERY_CACHE.clear()
    _ANCHOR_CACHE.clear()
    _CORPUS_FINGERPRINT_CACHE.clear()


def clear_query_cache():
    """Clear all process-local retrieval caches (useful for tests/reloads)."""
    clear_result_cache()
    _QUERY_EMBEDDING_CACHE.clear()
    for key in _QUERY_EMBEDDING_STATS:
        _QUERY_EMBEDDING_STATS[key] = 0
    for key in _QUERY_EMBEDDING_STATS:
        _QUERY_EMBEDDING_STATS[key] = 0


def query_cache_info():
    """Return bounded-cache sizes and cumulative embedding hit/miss counts."""
    return {"size": len(_QUERY_CACHE), "max_size": _CACHE_MAX_SIZE,
            "embedding_size": len(_QUERY_EMBEDDING_CACHE),
            "embedding_max_size": _QUERY_EMBEDDING_CACHE_MAX_SIZE,
            "embedding_cache_hits": _QUERY_EMBEDDING_STATS["hits"],
            "embedding_cache_misses": _QUERY_EMBEDDING_STATS["misses"],
            "embedding_cache_evictions": _QUERY_EMBEDDING_STATS["evictions"]}


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
        # A flat score vector contains no ranking signal. Returning ones here
        # (especially when every BM25 score is zero) gives unrelated
        # candidates full sparse relevance and can dominate hybrid fusion.
        return np.zeros_like(scores)

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
    Results are cached because adaptive retrieval revisits the same query at
    several candidate depths.
    """
    cache_key = (str(query), _corpus_fingerprint(texts), int(limit_per_focus))
    cached = _ANCHOR_CACHE.get(cache_key)
    if cached is not None:
        _ANCHOR_CACHE.move_to_end(cache_key)
        return list(cached)

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

    result = list(selected.keys())
    _ANCHOR_CACHE[cache_key] = tuple(result)
    _ANCHOR_CACHE.move_to_end(cache_key)
    while len(_ANCHOR_CACHE) > _ANCHOR_CACHE_MAX_SIZE:
        _ANCHOR_CACHE.popitem(last=False)
    return result


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
    return_candidates=False,
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

        return_candidates:
            False (default) returns top-k as before. True returns the
            complete fused candidate pool for downstream reranking.
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

    cache_key = _cache_key(
        query, faiss_index, bm25, texts, k, candidate_k,
        dense_weight, sparse_weight, fusion_method, rrf_k, return_candidates,
    )
    cached = _QUERY_CACHE.get(cache_key)
    if cached is not None:
        _QUERY_CACHE.move_to_end(cache_key)
        print("[Hybrid Retrieval] Query cache: HIT")
        cached_copy = copy.deepcopy(cached)
        cached_copy.setdefault("stage_timings_seconds", {})["query_result_cache_hit"] = True
        cached_copy["cache_diagnostics"] = query_cache_info()
        return cached_copy

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

    _embedding_started = perf_counter()
    query_embedding, embedding_cache_status = _get_query_embedding(query)
    embedding_seconds = perf_counter() - _embedding_started

    _dense_started = perf_counter()
    dense_scores, dense_ids = (
        faiss_index.search(
            query_embedding,
            candidate_k,
        )
    )
    dense_search_seconds = perf_counter() - _dense_started

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

    _bm25_started = perf_counter()
    bm25_tokens = tokenize(query)

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
    bm25_search_seconds = perf_counter() - _bm25_started

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
    _fusion_started = perf_counter()
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

    # By default retain the historical API: return only the requested top-k.
    # MS-ARC opts into the full candidate pool so CrossEncoder reranking can
    # rescue a strong candidate that hybrid fusion ranked just below k.
    output_limit = candidate_k if return_candidates else k
    merged_results = merged_results[:output_limit]

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

    fusion_seconds = perf_counter() - _fusion_started
    result = {
        "stage_timings_seconds": {
            "query_embedding_seconds": embedding_seconds,
            "dense_search_seconds": dense_search_seconds,
            "bm25_search_seconds": bm25_search_seconds,
            "candidate_fusion_seconds": fusion_seconds,
            "hybrid_search_total_seconds": embedding_seconds + dense_search_seconds + bm25_search_seconds + fusion_seconds,
            "query_result_cache_hit": False,
        },
        "embedding_cache_status": embedding_cache_status,
        "candidate_count": len(merged_results),
        "cache_diagnostics": query_cache_info(),
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

    _QUERY_CACHE[cache_key] = copy.deepcopy(result)
    _QUERY_CACHE.move_to_end(cache_key)
    while len(_QUERY_CACHE) > _CACHE_MAX_SIZE:
        _QUERY_CACHE.popitem(last=False)
    print("[Hybrid Retrieval] Query cache: STORE")
    return result