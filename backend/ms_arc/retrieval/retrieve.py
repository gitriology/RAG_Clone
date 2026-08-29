from __future__ import annotations

import os
import time
from pathlib import Path

import faiss

from backend.retrieval.utils.load_data import load_data
from backend.retrieval.index.faiss_index import build_faiss
from backend.retrieval.lexical.bm25 import (
    build_bm25,
    load_or_build_bm25,
)
from backend.retrieval.hybrid.hybrid_search import hybrid_search

from backend.ms_arc.state.retrieval_state import (
    RetrievalState,
    RetrievedDocument,
)


# ==========================================================
# STARTUP TIMING
# ==========================================================

_ENGINE_START_TIME = time.perf_counter()

BM25_FORCE_REBUILD = (
    os.getenv(
        "OPT14_FORCE_BM25_REBUILD",
        "0",
    ).strip().lower()
    in {
        "1",
        "true",
        "yes",
    }
)


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "all_domains.json"
)

FAISS_PATH = (
    PROJECT_ROOT
    / "backend"
    / "retrieval"
    / "index"
    / "faiss.index"
)

BM25_CACHE_PATH = (
    PROJECT_ROOT
    / "backend"
    / "retrieval"
    / "index"
    / "bm25_cache.pkl"
)


# ==========================================================
# DATASET
# ==========================================================

_dataset_start = time.perf_counter()

print("[MS-ARC] Loading Dataset...")

data = load_data(
    str(DATASET_PATH)
)

texts = [
    doc["text"]
    for doc in data
]

doc_lookup = {
    idx: doc
    for idx, doc in enumerate(data)
}

_dataset_time = (
    time.perf_counter()
    - _dataset_start
)

print(
    f"[MS-ARC] Dataset loaded in "
    f"{_dataset_time:.4f}s"
)


# ==========================================================
# EMBEDDINGS
# ==========================================================

_embeddings_start = time.perf_counter()

print(
    "[MS-ARC] Loading Cached Embeddings..."
)

from backend.retrieval.utils.load_embeddings import (
    load_embeddings,
)

embeddings = load_embeddings()

_embeddings_time = (
    time.perf_counter()
    - _embeddings_start
)

print(
    f"[MS-ARC] Embeddings loaded in "
    f"{_embeddings_time:.4f}s"
)


# ==========================================================
# FAISS
# ==========================================================

_faiss_start = time.perf_counter()

if FAISS_PATH.exists():

    print(
        "[MS-ARC] "
        "Loading Existing FAISS Index..."
    )

    faiss_index = faiss.read_index(
        str(FAISS_PATH)
    )

    faiss_action = "loaded"

else:

    print(
        "[MS-ARC] "
        "Building FAISS Index..."
    )

    faiss_index = build_faiss(
        embeddings
    )

    FAISS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        faiss_index,
        str(FAISS_PATH),
    )

    faiss_action = "built"

_faiss_time = (
    time.perf_counter()
    - _faiss_start
)

print(
    f"[MS-ARC] FAISS {faiss_action} in "
    f"{_faiss_time:.4f}s"
)

print(
    f"[MS-ARC] Dataset documents: "
    f"{len(data)}"
)

print(
    f"[MS-ARC] Text documents: "
    f"{len(texts)}"
)

print(
    f"[MS-ARC] FAISS vectors: "
    f"{faiss_index.ntotal}"
)


# ==========================================================
# BM25
# ==========================================================

_bm25_start = time.perf_counter()

print(
    "[MS-ARC] Initializing BM25..."
)

if BM25_FORCE_REBUILD:

    # ------------------------------------------------------
    # Optimization #14 benchmark baseline
    #
    # This reproduces the old behavior:
    #
    #     build_bm25(texts)
    #
    # No persistent cache is used.
    # ------------------------------------------------------

    print(
        "[MS-ARC] "
        "OPT14_FORCE_BM25_REBUILD=1"
    )

    print(
        "[MS-ARC] "
        "Forcing BM25 rebuild for baseline benchmark..."
    )

    bm25, tokenized = build_bm25(
        texts
    )

    bm25_loaded_from_cache = False

    bm25_mode = "forced_rebuild"

else:

    # ------------------------------------------------------
    # Optimization #14 optimized path
    #
    # First run:
    #
    #     build -> save
    #
    # Later runs:
    #
    #     load cache -> reuse
    # ------------------------------------------------------

    (
        bm25,
        tokenized,
        bm25_loaded_from_cache,
    ) = load_or_build_bm25(
        texts=texts,
        cache_path=BM25_CACHE_PATH,
    )

    if bm25_loaded_from_cache:

        bm25_mode = "cache"

    else:

        bm25_mode = "build_and_cache"


_bm25_time = (
    time.perf_counter()
    - _bm25_start
)

print(
    f"[MS-ARC] BM25 initialization "
    f"completed in {_bm25_time:.4f}s"
)

print(
    "[MS-ARC] BM25 mode:",
    bm25_mode,
)

print(
    "[MS-ARC] BM25 cache reused:",
    bm25_loaded_from_cache,
)

print(
    "[MS-ARC] BM25 cache path:",
    BM25_CACHE_PATH,
)


# ==========================================================
# ENGINE STARTUP SUMMARY
# ==========================================================

_ENGINE_STARTUP_TIME = (
    time.perf_counter()
    - _ENGINE_START_TIME
)

print()
print(
    "=================================================="
)

print(
    "MS-ARC STARTUP SUMMARY"
)

print(
    "=================================================="
)

print(
    f"Dataset loading:      "
    f"{_dataset_time:.4f}s"
)

print(
    f"Embedding loading:    "
    f"{_embeddings_time:.4f}s"
)

print(
    f"FAISS initialization: "
    f"{_faiss_time:.4f}s"
)

print(
    f"BM25 initialization:  "
    f"{_bm25_time:.4f}s"
)

print(
    f"Total engine startup:  "
    f"{_ENGINE_STARTUP_TIME:.4f}s"
)

print(
    f"BM25 mode:             "
    f"{bm25_mode}"
)

print(
    "=================================================="
)

print(
    "[MS-ARC] Retrieval Engine Ready."
)


# ==========================================================
# RETRIEVE
# ==========================================================

def retrieve(
    state: RetrievalState,
    fusion_method: str = "minmax",
) -> RetrievalState:

    """
    Performs hybrid retrieval.

    Supported fusion methods:

        minmax
        rrf
    """

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
            f"{fusion_method}"
        )

    print()
    print(
        "[MS-ARC] Hybrid Retrieval"
    )

    print(
        "[MS-ARC] Fusion method:",
        fusion_method,
    )

    candidate_k = state.debug.get(
        "candidate_k"
    )

    results = hybrid_search(

        query=state.query,

        faiss_index=faiss_index,

        bm25=bm25,

        texts=texts,

        k=state.recommended_topk,

        candidate_k=candidate_k,

        fusion_method=fusion_method,

    )

    dense_docs = []

    sparse_docs = []

    merged_docs = []


    # ======================================================
    # DENSE RESULTS
    # ======================================================

    for item in results[
        "dense_results"
    ]:

        original = doc_lookup[
            item["doc_id"]
        ]

        dense_docs.append(

            RetrievedDocument(

                doc_id=str(
                    item["doc_id"]
                ),

                text=item["text"],

                dense_score=float(
                    item["dense_score"]
                ),

                sparse_score=0.0,

                metadata={

                    "domain":
                        original.get(
                            "domain",
                            "general",
                        ),

                    "source":
                        original.get(
                            "source",
                            "unknown",
                        ),

                    "hybrid_score":
                        0.0,

                    "fusion_method":
                        fusion_method,

                    "dense_rank":
                        item.get(
                            "dense_rank"
                        ),

                    "sparse_rank":
                        None,

                },

            )

        )


    # ======================================================
    # SPARSE RESULTS
    # ======================================================

    for item in results[
        "sparse_results"
    ]:

        original = doc_lookup[
            item["doc_id"]
        ]

        sparse_docs.append(

            RetrievedDocument(

                doc_id=str(
                    item["doc_id"]
                ),

                text=item["text"],

                dense_score=0.0,

                sparse_score=float(
                    item["bm25_score"]
                ),

                metadata={

                    "domain":
                        original.get(
                            "domain",
                            "general",
                        ),

                    "source":
                        original.get(
                            "source",
                            "unknown",
                        ),

                    "hybrid_score":
                        0.0,

                    "fusion_method":
                        fusion_method,

                    "dense_rank":
                        None,

                    "sparse_rank":
                        item.get(
                            "sparse_rank"
                        ),

                },

            )

        )


    # ======================================================
    # HYBRID RESULTS
    # ======================================================

    for item in results[
        "merged_results"
    ]:

        original = doc_lookup[
            item["doc_id"]
        ]

        merged_docs.append(

            RetrievedDocument(

                doc_id=str(
                    item["doc_id"]
                ),

                text=item["text"],

                dense_score=float(
                    item["dense_score"]
                ),

                sparse_score=float(
                    item["bm25_score"]
                ),

                metadata={

                    "domain":
                        original.get(
                            "domain",
                            "general",
                        ),

                    "source":
                        original.get(
                            "source",
                            "unknown",
                        ),

                    "hybrid_score":
                        float(
                            item["hybrid_score"]
                        ),

                    "fusion_method":
                        fusion_method,

                    "dense_rank":
                        item.get(
                            "dense_rank"
                        ),

                    "sparse_rank":
                        item.get(
                            "sparse_rank"
                        ),

                },

            )

        )


    # ======================================================
    # STORE STATE
    # ======================================================

    state.dense_results = dense_docs

    state.sparse_results = sparse_docs

    state.merged_results = merged_docs

    state.selected_documents = (
        merged_docs
    )


    # ======================================================
    # DEBUG
    # ======================================================

    state.debug[
        "fusion_method"
    ] = fusion_method

    state.debug[
        "candidate_k_used"
    ] = results.get(
        "candidate_k"
    )

    state.debug[
        "bm25_cache_path"
    ] = str(
        BM25_CACHE_PATH
    )

    state.debug[
        "bm25_loaded_from_cache"
    ] = bm25_loaded_from_cache

    state.debug[
        "bm25_mode"
    ] = bm25_mode

    state.debug[
        "bm25_initialization_time"
    ] = _bm25_time

    state.debug[
        "engine_startup_time"
    ] = _ENGINE_STARTUP_TIME


    print(
        f"[MS-ARC] Retrieved "
        f"{len(merged_docs)} documents."
    )

    print(
        "[MS-ARC] Fusion method used:",
        fusion_method,
    )

    print(
        "[MS-ARC] BM25 mode:",
        bm25_mode,
    )

    return state


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    state = RetrievalState(
        query="What is WHO?"
    )

    state.recommended_topk = 5

    state = retrieve(
        state,
        fusion_method="minmax",
    )

    print()

    print(
        "=================================================="
    )

    print(
        "MS-ARC TEST RESULT"
    )

    print(
        "=================================================="
    )

    print(
        "Fusion:",
        state.debug.get(
            "fusion_method"
        ),
    )

    print(
        "BM25 mode:",
        state.debug.get(
            "bm25_mode"
        ),
    )

    print(
        "BM25 cache reused:",
        state.debug.get(
            "bm25_loaded_from_cache"
        ),
    )

    print(
        "BM25 initialization:",
        state.debug.get(
            "bm25_initialization_time"
        ),
        "seconds",
    )

    print(
        "Total engine startup:",
        state.debug.get(
            "engine_startup_time"
        ),
        "seconds",
    )

    print()

    for doc in state.selected_documents:

        print(
            doc.doc_id,
            doc.metadata.get(
                "fusion_method"
            ),
            doc.metadata.get(
                "hybrid_score"
            ),
        )