from __future__ import annotations

import argparse
from pathlib import Path

import faiss

from backend.retrieval.index.faiss_index import (
    DEFAULT_INDEX_TYPE,
    build_faiss,
    get_faiss_metadata,
)
from backend.retrieval.utils.load_embeddings import (
    load_embeddings,
)


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "backend"
    / "retrieval"
    / "index"
    / "faiss.index"
)


# ==========================================================
# BUILD
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build the FAISS retrieval index."
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
        default=str(
            DEFAULT_OUTPUT
        ),
        help=(
            "Output FAISS index path."
        ),
    )

    args = parser.parse_args()

    output = Path(
        args.output
    )

    print(
        "=" * 60
    )

    print(
        "FAISS INDEX BUILD"
    )

    print(
        "=" * 60
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Index type:   {args.index_type}"
    )

    print(
        f"Output:       {output}"
    )

    # ------------------------------------------------------
    # LOAD EMBEDDINGS
    # ------------------------------------------------------

    print()

    print(
        "[FAISS] Loading embeddings..."
    )

    embeddings = load_embeddings()

    print(
        f"[FAISS] Embedding shape: "
        f"{embeddings.shape}"
    )

    if embeddings.ndim != 2:

        raise ValueError(
            "Embeddings must be 2-dimensional. "
            f"Received: {embeddings.shape}"
        )

    # ------------------------------------------------------
    # BUILD
    # ------------------------------------------------------

    print()

    print(
        "[FAISS] Building index..."
    )

    index = build_faiss(
        embeddings,
        index_type=args.index_type,
    )

    # ------------------------------------------------------
    # FINAL CONSISTENCY CHECK
    # ------------------------------------------------------

    if index.ntotal != len(
        embeddings
    ):

        raise RuntimeError(
            "FAISS index contains a different "
            "number of vectors than the embeddings cache: "
            f"index={index.ntotal}, "
            f"embeddings={len(embeddings)}"
        )

    # ------------------------------------------------------
    # SAVE
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # VERIFY SAVED INDEX
    # ------------------------------------------------------

    print()

    print(
        "[FAISS] Re-opening saved index "
        "for verification..."
    )

    verified_index = faiss.read_index(
        str(output)
    )

    if verified_index.ntotal != len(
        embeddings
    ):

        raise RuntimeError(
            "Saved FAISS index failed "
            "the vector-count consistency check: "
            f"index={verified_index.ntotal}, "
            f"embeddings={len(embeddings)}"
        )

    if verified_index.d != embeddings.shape[1]:

        raise RuntimeError(
            "Saved FAISS index failed "
            "the dimension consistency check: "
            f"index={verified_index.d}, "
            f"embeddings={embeddings.shape[1]}"
        )

    metadata = get_faiss_metadata(
        verified_index
    )

    # ------------------------------------------------------
    # REPORT
    # ------------------------------------------------------

    print()

    print(
        "=" * 60
    )

    print(
        "FAISS INDEX BUILD COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Index type:          "
        f"{metadata.get('index_type')}"
    )

    print(
        f"Embedding dimension: "
        f"{metadata.get('dimension')}"
    )

    print(
        f"Indexed vectors:     "
        f"{metadata.get('ntotal')}"
    )

    print(
        f"Metric:              "
        f"{metadata.get('metric')}"
    )

    if (
        metadata.get(
            "index_type"
        )
        == "hnsw"
    ):

        print(
            f"HNSW M:              "
            f"{metadata.get('hnsw_m')}"
        )

        print(
            f"HNSW efSearch:       "
            f"{metadata.get('hnsw_ef_search')}"
        )

        print(
            f"HNSW efConstruction: "
            f"{metadata.get('hnsw_ef_construction')}"
        )

    print(
        f"Saved to:             "
        f"{output}"
    )

    print()

    print(
        "RESULT: PASS"
    )


if __name__ == "__main__":
    main()