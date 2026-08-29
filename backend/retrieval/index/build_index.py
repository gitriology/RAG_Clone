from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np

from backend.retrieval.utils.load_embeddings import load_embeddings


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "backend"
    / "retrieval"
    / "index"
    / "faiss.index"
)

DEFAULT_INDEX_TYPE = "hnsw"

# HNSW configuration
DEFAULT_HNSW_M = 32
DEFAULT_HNSW_EF_CONSTRUCTION = 200
DEFAULT_HNSW_EF_SEARCH = 64


# ============================================================
# HELPERS
# ============================================================

def _validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Validate and normalize the embedding array representation.

    We DO NOT normalize the vectors here because the existing
    embedding cache may already be normalized and changing the
    stored vectors during index construction would alter retrieval
    behaviour.

    FAISS requires float32 for normal CPU indexes.
    """

    if embeddings is None:
        raise ValueError("Embedding loader returned None.")

    embeddings = np.asarray(embeddings)

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2-dimensional array. "
            f"Received shape: {embeddings.shape}"
        )

    if embeddings.shape[0] == 0:
        raise ValueError("Embedding array is empty.")

    if embeddings.shape[1] <= 0:
        raise ValueError(
            "Embedding dimension must be greater than zero."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain NaN or infinite values."
        )

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(
            np.float32,
            copy=False,
        )

    return np.ascontiguousarray(embeddings)


def build_hnsw(
    embeddings: np.ndarray,
    m: int = DEFAULT_HNSW_M,
    ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION,
    ef_search: int = DEFAULT_HNSW_EF_SEARCH,
) -> faiss.Index:
    """
    Build an HNSW index using inner-product similarity.

    IMPORTANT:
    Do not attach arbitrary Python attributes to the FAISS
    object. FAISS SWIG wrappers reject attributes such as:

        index._rag_index_type = "hnsw"

    """

    dimension = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(
        dimension,
        m,
        faiss.METRIC_INNER_PRODUCT,
    )

    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search

    index.add(embeddings)

    return index


def build_flat(
    embeddings: np.ndarray,
) -> faiss.Index:
    """
    Build an exact inner-product FAISS index.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def build_index(
    embeddings: np.ndarray,
    index_type: str,
    hnsw_m: int = DEFAULT_HNSW_M,
    hnsw_ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION,
    hnsw_ef_search: int = DEFAULT_HNSW_EF_SEARCH,
) -> faiss.Index:
    """
    Construct the requested FAISS index.
    """

    index_type = index_type.lower().strip()

    if index_type == "hnsw":
        return build_hnsw(
            embeddings=embeddings,
            m=hnsw_m,
            ef_construction=hnsw_ef_construction,
            ef_search=hnsw_ef_search,
        )

    if index_type == "flat":
        return build_flat(embeddings)

    raise ValueError(
        f"Unsupported FAISS index type: {index_type!r}. "
        "Supported values: hnsw, flat."
    )


def get_index_type(index: faiss.Index) -> str:
    """
    Determine index type from the actual FAISS object.

    This avoids relying on custom attributes.
    """

    if isinstance(index, faiss.IndexHNSWFlat):
        return "hnsw"

    if isinstance(index, faiss.IndexFlatIP):
        return "flat"

    # Handle wrapped / derived FAISS types safely.
    class_name = type(index).__name__.lower()

    if "hnsw" in class_name:
        return "hnsw"

    if "flat" in class_name:
        return "flat"

    return class_name


def get_metric_name(index: faiss.Index) -> str:
    """
    Return the FAISS metric in human-readable form.
    """

    metric = getattr(
        index,
        "metric_type",
        None,
    )

    if metric == faiss.METRIC_INNER_PRODUCT:
        return "inner_product"

    if metric == faiss.METRIC_L2:
        return "l2"

    return str(metric)


def get_metadata(index: faiss.Index) -> dict:
    """
    Build metadata without modifying the FAISS object.
    """

    index_type = get_index_type(index)

    metadata = {
        "index_type": index_type,
        "dimension": int(index.d),
        "ntotal": int(index.ntotal),
        "metric": get_metric_name(index),
    }

    if index_type == "hnsw":
        hnsw = getattr(index, "hnsw", None)

        if hnsw is not None:
            metadata.update(
                {
                    "hnsw_m": int(
                        getattr(
                            hnsw,
                            "M",
                            DEFAULT_HNSW_M,
                        )
                    ),
                    "hnsw_ef_search": int(
                        getattr(
                            hnsw,
                            "efSearch",
                            DEFAULT_HNSW_EF_SEARCH,
                        )
                    ),
                    "hnsw_ef_construction": int(
                        getattr(
                            hnsw,
                            "efConstruction",
                            DEFAULT_HNSW_EF_CONSTRUCTION,
                        )
                    ),
                }
            )

    return metadata


def validate_index(
    index: faiss.Index,
    embeddings: np.ndarray,
) -> None:
    """
    Validate FAISS index against the embedding cache.
    """

    expected_vectors = embeddings.shape[0]
    expected_dimension = embeddings.shape[1]

    if index.ntotal != expected_vectors:
        raise RuntimeError(
            "FAISS vector-count mismatch.\n"
            f"Expected: {expected_vectors}\n"
            f"Actual:   {index.ntotal}"
        )

    if index.d != expected_dimension:
        raise RuntimeError(
            "FAISS dimension mismatch.\n"
            f"Expected: {expected_dimension}\n"
            f"Actual:   {index.d}"
        )

    metric = getattr(
        index,
        "metric_type",
        None,
    )

    if metric != faiss.METRIC_INNER_PRODUCT:
        raise RuntimeError(
            "FAISS metric mismatch.\n"
            "Expected: inner_product\n"
            f"Actual:   {metric}"
        )

    if not index.is_trained:
        raise RuntimeError(
            "FAISS index reports is_trained=False."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build and validate the production FAISS "
            "retrieval index."
        )
    )

    parser.add_argument(
        "--index-type",
        choices=[
            "hnsw",
            "flat",
        ],
        default=DEFAULT_INDEX_TYPE,
        help=(
            "FAISS index type. "
            "Default: hnsw"
        ),
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=(
            "Output FAISS index path."
        ),
    )

    parser.add_argument(
        "--hnsw-m",
        type=int,
        default=DEFAULT_HNSW_M,
        help=(
            "HNSW M parameter. "
            f"Default: {DEFAULT_HNSW_M}"
        ),
    )

    parser.add_argument(
        "--hnsw-ef-construction",
        type=int,
        default=DEFAULT_HNSW_EF_CONSTRUCTION,
        help=(
            "HNSW construction parameter. "
            f"Default: {DEFAULT_HNSW_EF_CONSTRUCTION}"
        ),
    )

    parser.add_argument(
        "--hnsw-ef-search",
        type=int,
        default=DEFAULT_HNSW_EF_SEARCH,
        help=(
            "HNSW search parameter. "
            f"Default: {DEFAULT_HNSW_EF_SEARCH}"
        ),
    )

    args = parser.parse_args()

    output = Path(args.output)

    print()
    print("=" * 64)
    print("FAISS INDEX BUILD")
    print("=" * 64)

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Index type:   {args.index_type}"
    )

    print(
        f"Output:       {output}"
    )

    # ========================================================
    # LOAD EMBEDDINGS
    # ========================================================

    print()
    print(
        "[FAISS] Loading embeddings..."
    )

    embeddings = load_embeddings()

    embeddings = _validate_embeddings(
        embeddings
    )

    print(
        f"[FAISS] Embedding shape: "
        f"{embeddings.shape}"
    )

    print(
        f"[FAISS] Embedding dtype: "
        f"{embeddings.dtype}"
    )

    print(
        f"[FAISS] Embedding vectors: "
        f"{len(embeddings)}"
    )

    print(
        f"[FAISS] Embedding dimension: "
        f"{embeddings.shape[1]}"
    )

    # ========================================================
    # BUILD INDEX
    # ========================================================

    print()
    print(
        "[FAISS] Building index..."
    )

    index = build_index(
        embeddings=embeddings,
        index_type=args.index_type,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
    )

    # ========================================================
    # VALIDATE IN-MEMORY INDEX
    # ========================================================

    print()
    print(
        "[FAISS] Validating in-memory index..."
    )

    validate_index(
        index,
        embeddings,
    )

    metadata = get_metadata(
        index
    )

    print(
        "[FAISS] In-memory consistency: PASS"
    )

    print(
        f"[FAISS] Index type: "
        f"{metadata['index_type']}"
    )

    print(
        f"[FAISS] Dimension: "
        f"{metadata['dimension']}"
    )

    print(
        f"[FAISS] Vectors: "
        f"{metadata['ntotal']}"
    )

    print(
        f"[FAISS] Metric: "
        f"{metadata['metric']}"
    )

    if metadata["index_type"] == "hnsw":

        print(
            f"[FAISS] HNSW M: "
            f"{metadata.get('hnsw_m')}"
        )

        print(
            f"[FAISS] HNSW efSearch: "
            f"{metadata.get('hnsw_ef_search')}"
        )

        print(
            f"[FAISS] HNSW efConstruction: "
            f"{metadata.get('hnsw_ef_construction')}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "[FAISS] Saving index..."
    )

    faiss.write_index(
        index,
        str(output),
    )

    print(
        f"[FAISS] Saved: {output}"
    )

    # ========================================================
    # VERIFY SAVED INDEX
    # ========================================================

    print()
    print(
        "[FAISS] Re-opening saved index "
        "for verification..."
    )

    verified_index = faiss.read_index(
        str(output)
    )

    validate_index(
        verified_index,
        embeddings,
    )

    verified_metadata = get_metadata(
        verified_index
    )

    print(
        "[FAISS] Saved index consistency: PASS"
    )

    # ========================================================
    # EXTRA SEARCH SANITY CHECK
    # ========================================================

    print()
    print(
        "[FAISS] Running search sanity check..."
    )

    test_k = min(
        3,
        len(embeddings),
    )

    query_vector = embeddings[0:1]

    distances, indices = verified_index.search(
        query_vector,
        test_k,
    )

    if indices.shape != (1, test_k):
        raise RuntimeError(
            "FAISS search returned an unexpected "
            f"shape: {indices.shape}"
        )

    if np.any(indices < 0):
        raise RuntimeError(
            "FAISS search returned invalid "
            "negative document IDs."
        )

    if not np.isfinite(distances).all():
        raise RuntimeError(
            "FAISS search returned NaN or "
            "infinite similarity scores."
        )

    print(
        "[FAISS] Search sanity check: PASS"
    )

    print(
        f"[FAISS] Test query top-{test_k} IDs: "
        f"{indices[0].tolist()}"
    )

    print(
        f"[FAISS] Test query scores: "
        f"{distances[0].tolist()}"
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 64)
    print("FAISS INDEX BUILD COMPLETE")
    print("=" * 64)

    print(
        f"Index type:          "
        f"{verified_metadata.get('index_type')}"
    )

    print(
        f"Embedding dimension: "
        f"{verified_metadata.get('dimension')}"
    )

    print(
        f"Indexed vectors:     "
        f"{verified_metadata.get('ntotal')}"
    )

    print(
        f"Metric:              "
        f"{verified_metadata.get('metric')}"
    )

    if (
        verified_metadata.get(
            "index_type"
        )
        == "hnsw"
    ):

        print(
            f"HNSW M:              "
            f"{verified_metadata.get('hnsw_m')}"
        )

        print(
            f"HNSW efSearch:       "
            f"{verified_metadata.get('hnsw_ef_search')}"
        )

        print(
            f"HNSW efConstruction: "
            f"{verified_metadata.get('hnsw_ef_construction')}"
        )

    print(
        f"Saved to:             "
        f"{output}"
    )

    print()
    print(
        "RESULT: PASS"
    )

    print()


if __name__ == "__main__":
    main()