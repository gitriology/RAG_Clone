from __future__ import annotations

from typing import Any, Dict, Tuple

import faiss
import numpy as np


# ==========================================================
# INDEX CONFIGURATION
# ==========================================================

DEFAULT_HNSW_M = 64
DEFAULT_HNSW_EF_SEARCH = 32
DEFAULT_HNSW_EF_CONSTRUCTION = 40


# ==========================================================
# BUILD FAISS INDEX
# ==========================================================

def build_faiss(
    embeddings,
    index_type: str = "flat",
    hnsw_m: int = DEFAULT_HNSW_M,
    hnsw_ef_search: int = DEFAULT_HNSW_EF_SEARCH,
    hnsw_ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION,
):
    """
    Build a FAISS inner-product index.

    Supported index types:

        flat
        hnsw

    Embeddings are expected to already be L2-normalized.
    """

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2D array. "
            f"Received shape: {embeddings.shape}"
        )

    if embeddings.shape[0] == 0:
        raise ValueError(
            "Cannot build FAISS index from zero embeddings."
        )

    dimension = embeddings.shape[1]

    index_type = (
        index_type
        or "flat"
    ).lower().strip()

    print(
        f"[FAISS] Index type             : {index_type}"
    )

    print(
        f"[FAISS] Embedding dimension    : {dimension}"
    )

    print(
        f"[FAISS] Embedding count        : "
        f"{len(embeddings)}"
    )

    # ------------------------------------------------------
    # FLAT
    # ------------------------------------------------------

    if index_type == "flat":

        index = faiss.IndexFlatIP(
            dimension
        )

    # ------------------------------------------------------
    # HNSW
    # ------------------------------------------------------

    elif index_type == "hnsw":

        index = faiss.IndexHNSWFlat(
            dimension,
            hnsw_m,
            faiss.METRIC_INNER_PRODUCT,
        )

        index.hnsw.efSearch = (
            hnsw_ef_search
        )

        index.hnsw.efConstruction = (
            hnsw_ef_construction
        )

    else:

        raise ValueError(
            "Unsupported FAISS index type: "
            f"{index_type}. "
            "Supported types: flat, hnsw."
        )

    # ------------------------------------------------------
    # ADD VECTORS
    # ------------------------------------------------------

    index.add(
        embeddings
    )

    print(
        f"[FAISS] Indexed vectors       : "
        f"{index.ntotal}"
    )

    return index


# ==========================================================
# SEARCH
# ==========================================================

def search_faiss(
    query_embedding,
    index,
    k: int = 5,
):
    """
    Search the FAISS index.

    Returns
    -------
    scores:
        Inner-product similarity scores.

    indices:
        FAISS row indices.

    Important:
        FAISS indices are dataset ROW indices in this
        project. They must not automatically be treated
        as the dataset's `id` field because dataset IDs
        are not globally unique.
    """

    if k <= 0:
        return (
            np.asarray(
                [],
                dtype=np.float32,
            ),
            np.asarray(
                [],
                dtype=np.int64,
            ),
        )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    # ------------------------------------------------------
    # Normalize query shape
    # ------------------------------------------------------

    if query_embedding.ndim == 1:

        query_embedding = (
            query_embedding.reshape(
                1,
                -1,
            )
        )

    if query_embedding.ndim != 2:
        raise ValueError(
            "Query embedding must be 1D or 2D. "
            f"Received shape: {query_embedding.shape}"
        )

    if query_embedding.shape[0] != 1:
        raise ValueError(
            "search_faiss expects exactly one "
            "query embedding."
        )

    if query_embedding.shape[1] != index.d:
        raise ValueError(
            "Query embedding dimension does not "
            "match FAISS index dimension. "
            f"Query={query_embedding.shape[1]}, "
            f"Index={index.d}"
        )

    actual_k = min(
        int(k),
        int(index.ntotal),
    )

    if actual_k <= 0:
        return (
            np.asarray(
                [],
                dtype=np.float32,
            ),
            np.asarray(
                [],
                dtype=np.int64,
            ),
        )

    scores, indices = index.search(
        query_embedding,
        actual_k,
    )

    return (
        scores[0],
        indices[0],
    )


# ==========================================================
# INDEX METADATA
# ==========================================================

def get_index_metadata(
    index,
) -> Dict[str, Any]:
    """
    Return metadata describing the FAISS index.

    This is used by the Optimization #15 benchmark and
    production diagnostics.

    The metadata intentionally records FAISS ROW capacity
    (`ntotal`) rather than assuming that FAISS row numbers
    equal dataset `id` values.
    """

    metadata: Dict[str, Any] = {
        "index_type": "unknown",
        "dimension": int(
            index.d
        ),
        "ntotal": int(
            index.ntotal
        ),
        "metric": "unknown",
    }

    # ------------------------------------------------------
    # Metric
    # ------------------------------------------------------

    metric_type = getattr(
        index,
        "metric_type",
        None,
    )

    if metric_type == faiss.METRIC_INNER_PRODUCT:

        metadata["metric"] = (
            "inner_product"
        )

    elif metric_type == faiss.METRIC_L2:

        metadata["metric"] = "l2"

    # ------------------------------------------------------
    # Index type
    # ------------------------------------------------------

    if isinstance(
        index,
        faiss.IndexFlatIP,
    ):

        metadata["index_type"] = "flat"

    elif isinstance(
        index,
        faiss.IndexHNSWFlat,
    ):

        metadata["index_type"] = "hnsw"

        metadata["hnsw_m"] = int(
            index.hnsw.nb_neighbors(0)
        )

        metadata["hnsw_ef_search"] = int(
            index.hnsw.efSearch
        )

        metadata["hnsw_ef_construction"] = int(
            index.hnsw.efConstruction
        )

    return metadata
def get_faiss_metadata(index) -> Dict[str, Any]:
    """
    Backward-compatible wrapper for production retrieval code.

    The canonical metadata function is get_index_metadata().
    """
    return get_index_metadata(index)

# ==========================================================
# INDEX VALIDATION
# ==========================================================

def validate_faiss_index(
    index,
    embeddings=None,
) -> Tuple[bool, str]:
    """
    Validate basic consistency between a FAISS index and
    the embedding matrix.

    Checks:

        - index exists
        - embedding dimensionality matches
        - vector count matches embeddings
        - metric is inner product

    Returns:

        (True, "PASS")

    or:

        (False, reason)
    """

    if index is None:

        return (
            False,
            "FAISS index is None.",
        )

    if embeddings is not None:

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:

            return (
                False,
                "Embeddings are not a 2D matrix.",
            )

        if index.d != embeddings.shape[1]:

            return (
                False,
                "FAISS dimension does not match "
                "embedding dimension: "
                f"index={index.d}, "
                f"embeddings={embeddings.shape[1]}",
            )

        if index.ntotal != len(
            embeddings
        ):

            return (
                False,
                "FAISS vector count does not match "
                "embedding rows: "
                f"index={index.ntotal}, "
                f"embeddings={len(embeddings)}",
            )

    metric_type = getattr(
        index,
        "metric_type",
        None,
    )

    if metric_type != faiss.METRIC_INNER_PRODUCT:

        return (
            False,
            "FAISS metric is not inner product.",
        )

    return (
        True,
        "PASS",
    )