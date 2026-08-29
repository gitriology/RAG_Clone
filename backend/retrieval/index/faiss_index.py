from __future__ import annotations

import numpy as np
import faiss


# ==========================================================
# CONFIGURATION
# ==========================================================

DEFAULT_INDEX_TYPE = "flat"

SUPPORTED_INDEX_TYPES = {
    "flat",
    "hnsw",
}


# ==========================================================
# VALIDATION
# ==========================================================

def _prepare_embeddings(embeddings) -> np.ndarray:
    """
    Convert embeddings to float32 numpy array.

    Expected shape:

        (number_of_documents, embedding_dimension)
    """

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2D array with shape "
            "(num_documents, embedding_dimension)."
        )

    if embeddings.shape[0] == 0:
        raise ValueError(
            "Cannot build FAISS index from zero embeddings."
        )

    if embeddings.shape[1] == 0:
        raise ValueError(
            "Embedding dimension cannot be zero."
        )

    return np.ascontiguousarray(
        embeddings,
        dtype=np.float32,
    )


# ==========================================================
# BUILD FAISS INDEX
# ==========================================================

def build_faiss(
    embeddings,
    index_type: str = DEFAULT_INDEX_TYPE,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 40,
    hnsw_ef_search: int = 32,
):
    """
    Build a FAISS index.

    Supported index types:

        flat
            Exact Inner Product search.

        hnsw
            Approximate HNSW Inner Product search.

    The project uses normalized BGE embeddings, therefore
    Inner Product corresponds to cosine similarity.

    Parameters
    ----------
    embeddings:
        Normalized document embeddings.

    index_type:
        "flat" or "hnsw".

    hnsw_m:
        HNSW graph connectivity parameter.

    hnsw_ef_construction:
        HNSW construction-time search depth.

    hnsw_ef_search:
        HNSW query-time search depth.
    """

    index_type = (
        index_type
        or DEFAULT_INDEX_TYPE
    ).lower().strip()

    if index_type not in SUPPORTED_INDEX_TYPES:
        raise ValueError(
            f"Unsupported FAISS index type: "
            f"{index_type}. "
            f"Expected one of "
            f"{sorted(SUPPORTED_INDEX_TYPES)}."
        )

    embeddings = _prepare_embeddings(
        embeddings
    )

    dimension = embeddings.shape[1]

    print(
        f"[FAISS] Index type        : {index_type}"
    )

    print(
        f"[FAISS] Embedding dimension: {dimension}"
    )

    print(
        f"[FAISS] Embedding count   : "
        f"{len(embeddings)}"
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

    else:

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

    # ------------------------------------------------------
    # ADD VECTORS
    # ------------------------------------------------------

    index.add(
        embeddings
    )

    print(
        f"[FAISS] Indexed vectors  : "
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
    Search a FAISS index.

    Returns
    -------

    scores:
        Similarity scores.

    indices:
        Document ids.

    For normalized embeddings using Inner Product,
    scores correspond to cosine similarity.
    """

    if k <= 0:
        return (
            np.array(
                [],
                dtype=np.float32,
            ),
            np.array(
                [],
                dtype=np.int64,
            ),
        )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    if query_embedding.ndim == 1:
        query_embedding = (
            query_embedding.reshape(
                1,
                -1,
            )
        )

    if query_embedding.ndim != 2:
        raise ValueError(
            "Query embedding must be a 1D or 2D array."
        )

    query_embedding = np.ascontiguousarray(
        query_embedding,
        dtype=np.float32,
    )

    if query_embedding.shape[1] != index.d:
        raise ValueError(
            f"Query embedding dimension "
            f"{query_embedding.shape[1]} does not match "
            f"FAISS dimension {index.d}."
        )

    # HNSW has a query-time efSearch parameter.
    # Only set it when the index exposes HNSW.
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = max(
            int(k),
            int(getattr(
                index.hnsw,
                "efSearch",
                32,
            )),
        )

    actual_k = min(
        int(k),
        int(index.ntotal),
    )

    if actual_k <= 0:
        return (
            np.array(
                [],
                dtype=np.float32,
            ),
            np.array(
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

def get_index_type(index) -> str:
    """
    Return a human-readable FAISS index type.
    """

    if isinstance(
        index,
        faiss.IndexFlatIP,
    ):
        return "flat"

    if isinstance(
        index,
        faiss.IndexHNSWFlat,
    ):
        return "hnsw"

    return type(index).__name__


def get_index_metadata(index) -> dict:
    """
    Return useful metadata for benchmarking/debugging.
    """

    metadata = {
        "index_type": get_index_type(
            index
        ),
        "dimension": int(
            index.d
        ),
        "ntotal": int(
            index.ntotal
        ),
        "metric": "inner_product",
    }

    if hasattr(index, "hnsw"):

        metadata.update({
            "hnsw_m": int(
                index.hnsw.nb_neighbors(0)
            ),
            "hnsw_ef_search": int(
                index.hnsw.efSearch
            ),
            "hnsw_ef_construction": int(
                index.hnsw.efConstruction
            ),
        })

    return metadata