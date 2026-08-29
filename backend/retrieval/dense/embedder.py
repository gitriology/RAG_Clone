from __future__ import annotations

from typing import Iterable, List

import numpy as np

from backend.models.model_registry import ModelRegistry


# ==========================================================
# EMBEDDING CONFIGURATION
# ==========================================================

DEFAULT_BATCH_SIZE = 32


# ==========================================================
# DOCUMENT EMBEDDINGS
# ==========================================================

def get_embeddings(
    texts: Iterable[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """
    Generate normalized document embeddings.

    Parameters
    ----------
    texts:
        Iterable of document texts.

    batch_size:
        Batch size passed to SentenceTransformer.encode().

    Returns
    -------
    np.ndarray
        Float32 embedding matrix with shape:

            (number_of_texts, embedding_dimension)

    Notes
    -----
    The embedding model is obtained from ModelRegistry so that
    the project uses one shared BGE model instance.
    """

    texts = list(texts)

    if not texts:
        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    model = ModelRegistry.get_embedding_model()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return np.asarray(
        embeddings,
        dtype=np.float32,
    )


# ==========================================================
# BATCH QUERY EMBEDDINGS
# ==========================================================

def encode_queries(
    queries: Iterable[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """
    Generate normalized embeddings for multiple queries
    in a single model.encode() call.

    This is the main API introduced for Optimization #16.

    Parameters
    ----------
    queries:
        Iterable of query strings.

    batch_size:
        Batch size passed to SentenceTransformer.encode().

    Returns
    -------
    np.ndarray
        Float32 embedding matrix with shape:

            (number_of_queries, embedding_dimension)

    Example
    -------
    queries = [
        "what is machine learning?",
        "what is WHO?",
        "what is ISRO?",
    ]

    embeddings = encode_queries(queries)

    Result shape:

        (3, 768)

    Important
    ---------
    All queries are encoded by ONE model.encode() call.

    This avoids repeatedly invoking the embedding model for
    multiple queries.
    """

    queries = list(queries)

    if not queries:
        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    # ------------------------------------------------------
    # Validate query values
    # ------------------------------------------------------

    for index, query in enumerate(queries):

        if not isinstance(query, str):

            raise TypeError(
                "All queries must be strings. "
                f"Query at index {index} has type "
                f"{type(query).__name__}."
            )

    # ------------------------------------------------------
    # CENTRALIZED MODEL
    # ------------------------------------------------------

    model = ModelRegistry.get_embedding_model()

    # ------------------------------------------------------
    # SINGLE BATCH INFERENCE
    # ------------------------------------------------------

    embeddings = model.encode(
        queries,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    # ------------------------------------------------------
    # SHAPE VALIDATION
    # ------------------------------------------------------

    if embeddings.ndim != 2:

        raise ValueError(
            "Batch query encoding returned an unexpected "
            f"shape: {embeddings.shape}"
        )

    if embeddings.shape[0] != len(queries):

        raise ValueError(
            "Number of query embeddings does not match "
            "number of queries. "
            f"Queries={len(queries)}, "
            f"Embeddings={embeddings.shape[0]}"
        )

    return embeddings


# ==========================================================
# SINGLE QUERY EMBEDDING
# ==========================================================

def encode_query(
    query: str,
) -> np.ndarray:
    """
    Generate a normalized embedding for one query.

    This function is intentionally retained for compatibility
    with the existing production retrieval pipeline.

    Internally it delegates to encode_queries().

    Returns
    -------
    np.ndarray
        Shape:

            (1, embedding_dimension)

    This preserves the existing behavior expected by the
    FAISS retrieval code.
    """

    if not isinstance(query, str):

        raise TypeError(
            "query must be a string."
        )

    return encode_queries(
        [query],
        batch_size=1,
    )