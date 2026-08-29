from __future__ import annotations

from typing import Any

import faiss
import numpy as np


# ==========================================================
# CONFIGURATION
# ==========================================================

DEFAULT_INDEX_TYPE = "hnsw"

SUPPORTED_INDEX_TYPES = {
    "flat",
    "hnsw",
}


# ==========================================================
# VALIDATION
# ==========================================================

def _prepare_embeddings(embeddings: Any) -> np.ndarray:
    """
    Convert embeddings into a contiguous float32 matrix.

    Expected shape:
        (num_documents, embedding_dimension)
    """

    array = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if array.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2-dimensional array. "
            f"Received shape={array.shape}"
        )

    if array.shape[0] == 0:
        raise ValueError(
            "Cannot build FAISS index from zero embeddings."
        )

    if array.shape[1] == 0:
        raise ValueError(
            "Embedding dimension cannot be zero."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "Embeddings contain NaN or infinite values."
        )

    return np.ascontiguousarray(
        array,
        dtype=np.float32,
    )


# ==========================================================
# BUILD
# ==========================================================

def build_faiss(
    embeddings: Any,
    index_type: str = DEFAULT_INDEX_TYPE,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 40,
    hnsw_ef_search: int = 32,
):
    """
    Build a FAISS index using inner-product similarity.

    The project uses normalized BGE embeddings, so inner product
    corresponds to cosine similarity.
    """

    embeddings = _prepare_embeddings(
        embeddings
    )

    index_type = str(
        index_type
    ).strip().lower()

    if index_type not in SUPPORTED_INDEX_TYPES:
        raise ValueError(
            f"Unsupported FAISS index type: {index_type!r}. "
            f"Supported types: {sorted(SUPPORTED_INDEX_TYPES)}"
        )

    dimension = int(
        embeddings.shape[1]
    )

    # ------------------------------------------------------
    # EXACT FLAT INDEX
    # ------------------------------------------------------

    if index_type == "flat":

        index = faiss.IndexFlatIP(
            dimension
        )

    # ------------------------------------------------------
    # HNSW INDEX
    # ------------------------------------------------------

    elif index_type == "hnsw":

        if hnsw_m <= 0:
            raise ValueError(
                "hnsw_m must be greater than zero."
            )

        if hnsw_ef_construction <= 0:
            raise ValueError(
                "hnsw_ef_construction must be greater than zero."
            )

        if hnsw_ef_search <= 0:
            raise ValueError(
                "hnsw_ef_search must be greater than zero."
            )

        index = faiss.IndexHNSWFlat(
            dimension,
            int(hnsw_m),
            faiss.METRIC_INNER_PRODUCT,
        )

        index.hnsw.efConstruction = int(
            hnsw_ef_construction
        )

        index.hnsw.efSearch = int(
            hnsw_ef_search
        )

    else:
        raise RuntimeError(
            "Unreachable FAISS index type."
        )

    # ------------------------------------------------------
    # ADD VECTORS
    # ------------------------------------------------------

    index.add(
        embeddings
    )

    # ------------------------------------------------------
    # ATTACH METADATA
    #
    # FAISS itself does not persist arbitrary Python metadata,
    # but these attributes are useful while the object is alive.
    # The persisted index is still validated through its actual
    # FAISS properties.
    # ------------------------------------------------------

    index._rag_index_type = index_type
    index._rag_dimension = dimension
    index._rag_ntotal = int(
        index.ntotal
    )

    if index_type == "hnsw":
        index._rag_hnsw_m = int(
            hnsw_m
        )
        index._rag_hnsw_ef_construction = int(
            hnsw_ef_construction
        )
        index._rag_hnsw_ef_search = int(
            hnsw_ef_search
        )

    return index


# ==========================================================
# SEARCH
# ==========================================================

def search_faiss(
    query_embedding: Any,
    index,
    k: int = 5,
):
    """
    Search FAISS using inner-product similarity.

    Returns:
        scores, indices
    """

    if index is None:
        raise ValueError(
            "FAISS index cannot be None."
        )

    if k <= 0:
        return (
            np.empty(
                (0,),
                dtype=np.float32,
            ),
            np.empty(
                (0,),
                dtype=np.int64,
            ),
        )

    query = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    # Accept:
    #   (dimension,)
    # or:
    #   (1, dimension)

    if query.ndim == 1:
        query = query.reshape(
            1,
            -1,
        )

    if query.ndim != 2:
        raise ValueError(
            "Query embedding must be 1D or 2D. "
            f"Received shape={query.shape}"
        )

    if query.shape[1] != index.d:
        raise ValueError(
            "Query/index dimension mismatch. "
            f"Query dimension={query.shape[1]}, "
            f"index dimension={index.d}"
        )

    if not np.isfinite(query).all():
        raise ValueError(
            "Query embedding contains NaN or infinite values."
        )

    query = np.ascontiguousarray(
        query,
        dtype=np.float32,
    )

    actual_k = min(
        int(k),
        int(index.ntotal),
    )

    if actual_k <= 0:
        return (
            np.empty(
                (query.shape[0], 0),
                dtype=np.float32,
            ),
            np.empty(
                (query.shape[0], 0),
                dtype=np.int64,
            ),
        )

    scores, indices = index.search(
        query,
        actual_k,
    )

    return (
        scores,
        indices,
    )


# ==========================================================
# METADATA
# ==========================================================

def get_faiss_metadata(index) -> dict:
    """
    Return metadata derived from the actual FAISS index.

    Do not trust stale sidecar metadata for dimension, count,
    or metric.
    """

    if index is None:
        raise ValueError(
            "FAISS index cannot be None."
        )

    metadata = {
        "index_type": _detect_index_type(
            index
        ),
        "dimension": int(
            index.d
        ),
        "ntotal": int(
            index.ntotal
        ),
        "metric": _detect_metric(
            index
        ),
    }

    if hasattr(index, "hnsw"):
        metadata.update({
            "hnsw_m": int(
                index.hnsw.nb_neighbors(0)
                if index.ntotal > 0
                else 0
            ),
            "hnsw_ef_search": int(
                index.hnsw.efSearch
            ),
            "hnsw_ef_construction": int(
                index.hnsw.efConstruction
            ),
        })

    return metadata


# Backwards-compatible alias.
get_index_metadata = get_faiss_metadata


def _detect_index_type(index) -> str:

    if isinstance(
        index,
        faiss.IndexHNSW,
    ):
        return "hnsw"

    if isinstance(
        index,
        faiss.IndexFlat,
    ):
        return "flat"

    # Handle wrapped indexes.
    description = (
        type(index).__name__
        .lower()
    )

    if "hnsw" in description:
        return "hnsw"

    if "flat" in description:
        return "flat"

    return description


def _detect_metric(index) -> str:

    metric_type = getattr(
        index,
        "metric_type",
        None,
    )

    if metric_type == faiss.METRIC_INNER_PRODUCT:
        return "inner_product"

    if metric_type == faiss.METRIC_L2:
        return "l2"

    return str(
        metric_type
    )


# ==========================================================
# CONSISTENCY CHECK
# ==========================================================

def validate_faiss_consistency(
    index,
    embeddings,
) -> dict:
    """
    Verify that the persisted FAISS index matches the
    embedding matrix.
    """

    embeddings = _prepare_embeddings(
        embeddings
    )

    expected_count = int(
        embeddings.shape[0]
    )

    expected_dimension = int(
        embeddings.shape[1]
    )

    actual_count = int(
        index.ntotal
    )

    actual_dimension = int(
        index.d
    )

    count_ok = (
        actual_count
        == expected_count
    )

    dimension_ok = (
        actual_dimension
        == expected_dimension
    )

    passed = (
        count_ok
        and dimension_ok
    )

    result = {
        "passed": passed,
        "count_ok": count_ok,
        "dimension_ok": dimension_ok,
        "embedding_count": expected_count,
        "faiss_vectors": actual_count,
        "embedding_dimension": expected_dimension,
        "faiss_dimension": actual_dimension,
        "index_type": _detect_index_type(
            index
        ),
        "metric": _detect_metric(
            index
        ),
    }

    if not passed:
        raise RuntimeError(
            "FAISS consistency check failed.\n"
            f"Embedding count: {expected_count}\n"
            f"FAISS vectors: {actual_count}\n"
            f"Embedding dimension: {expected_dimension}\n"
            f"FAISS dimension: {actual_dimension}"
        )

    return result


# ==========================================================
# PERSISTENCE HELPERS
# ==========================================================

def save_faiss(
    index,
    path,
) -> None:

    path = str(
        path
    )

    faiss.write_index(
        index,
        path,
    )


def load_faiss(
    path,
):
    return faiss.read_index(
        str(path)
    )