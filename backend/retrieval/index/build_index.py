from __future__ import annotations

import argparse
import time
from pathlib import Path

import faiss

from backend.retrieval.utils.load_embeddings import (
    load_embeddings,
)

from backend.retrieval.index.faiss_index import (
    build_faiss,
    get_index_metadata,
)


# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]

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
            "Build the project's FAISS retrieval index."
        )
    )

    parser.add_argument(
        "--index-type",
        choices=[
            "flat",
            "hnsw",
        ],
        default="flat",
        help=(
            "FAISS index type. "
            "Default: flat"
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

    parser.add_argument(
        "--hnsw-m",
        type=int,
        default=32,
        help=(
            "HNSW M parameter. "
            "Default: 32"
        ),
    )

    parser.add_argument(
        "--hnsw-ef-construction",
        type=int,
        default=40,
        help=(
            "HNSW construction parameter. "
            "Default: 40"
        ),
    )

    parser.add_argument(
        "--hnsw-ef-search",
        type=int,
        default=32,
        help=(
            "HNSW search parameter. "
            "Default: 32"
        ),
    )

    args = parser.parse_args()

    print()
    print(
        "=================================================="
    )
    print(
        "FAISS INDEX BUILD"
    )
    print(
        "=================================================="
    )

    # ------------------------------------------------------
    # LOAD EMBEDDINGS
    # ------------------------------------------------------

    print(
        "[FAISS] Loading cached embeddings..."
    )

    embeddings = load_embeddings()

    print(
        f"[FAISS] Embeddings shape: "
        f"{embeddings.shape}"
    )

    # ------------------------------------------------------
    # BUILD
    # ------------------------------------------------------

    print()
    print(
        f"[FAISS] Building "
        f"{args.index_type.upper()} index..."
    )

    start = time.perf_counter()

    index = build_faiss(

        embeddings,

        index_type=args.index_type,

        hnsw_m=args.hnsw_m,

        hnsw_ef_construction=(
            args.hnsw_ef_construction
        ),

        hnsw_ef_search=(
            args.hnsw_ef_search
        ),

    )

    build_time = (
        time.perf_counter()
        - start
    )

    # ------------------------------------------------------
    # SAVE
    # ------------------------------------------------------

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(output),
    )

    # ------------------------------------------------------
    # RESULT
    # ------------------------------------------------------

    metadata = get_index_metadata(
        index
    )

    print()
    print(
        "=================================================="
    )

    print(
        "FAISS INDEX CREATED"
    )

    print(
        "=================================================="
    )

    print(
        "Index type:",
        metadata["index_type"],
    )

    print(
        "Dimension:",
        metadata["dimension"],
    )

    print(
        "Vectors:",
        metadata["ntotal"],
    )

    print(
        f"Build time: "
        f"{build_time:.4f}s"
    )

    print(
        "Saved to:",
        output,
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()