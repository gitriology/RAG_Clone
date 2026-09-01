from __future__ import annotations

from pathlib import Path

import faiss

from backend.retrieval.utils.load_data import (
    load_data,
)

from backend.retrieval.index.faiss_index import (
    build_faiss,
    get_faiss_metadata,
)

from backend.retrieval.lexical.bm25 import (
    load_or_build_bm25,
)

from backend.retrieval.hybrid.hybrid_search import (
    hybrid_search,
)

from backend.ms_arc.state.retrieval_state import (
    RetrievalState,
    RetrievedDocument,
)


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

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

print(
    "[MS-ARC] Loading Dataset..."
)

data = load_data(
    str(DATASET_PATH)
)

texts = [
    doc["text"]
    for doc in data
]

# ----------------------------------------------------------
# IMPORTANT:
#
# FAISS returns ROW INDICES.
#
# The dataset `id` field is not globally unique.
#
# Therefore:
#
#     FAISS row 1885
#
# means:
#
#     data[1885]
#
# and should not be interpreted as:
#
#     dataset id == 1885
#
# ----------------------------------------------------------

doc_lookup = {
    idx: doc
    for idx, doc in enumerate(data)
}


# ==========================================================
# EMBEDDINGS
# ==========================================================

print(
    "[MS-ARC] Loading Cached Embeddings..."
)

from backend.retrieval.utils.load_embeddings import (
    load_embeddings,
)

embeddings = load_embeddings()

print(
    f"[MS-ARC] Embeddings shape: "
    f"{embeddings.shape}"
)

if embeddings.ndim != 2:

    raise ValueError(
        "Embeddings must be a 2-dimensional "
        f"array. Received {embeddings.shape}"
    )


# ==========================================================
# FAISS
# ==========================================================

def _load_or_build_faiss():

    # ------------------------------------------------------
    # EXISTING PERSISTED INDEX
    # ------------------------------------------------------

    if FAISS_PATH.exists():

        print(
            "[MS-ARC] Loading Existing FAISS Index..."
        )

        index = faiss.read_index(
            str(FAISS_PATH)
        )

        metadata = get_faiss_metadata(
            index
        )

        print(
            "[MS-ARC] FAISS index type:",
            metadata.get(
                "index_type"
            ),
        )

        print(
            "[MS-ARC] FAISS dimension:",
            metadata.get(
                "dimension"
            ),
        )

        print(
            "[MS-ARC] FAISS vectors:",
            metadata.get(
                "ntotal"
            ),
        )

        # --------------------------------------------------
        # EMBEDDING / INDEX DIMENSION CHECK
        # --------------------------------------------------

        if index.d != embeddings.shape[1]:

            raise RuntimeError(
                "FAISS embedding dimension mismatch.\n"
                f"FAISS index dimension: {index.d}\n"
                f"Embedding dimension: {embeddings.shape[1]}\n"
                "Rebuild the FAISS index."
            )

        # --------------------------------------------------
        # EMBEDDING / INDEX COUNT CHECK
        # --------------------------------------------------

        if index.ntotal != len(
            embeddings
        ):

            raise RuntimeError(
                "FAISS vector count mismatch.\n"
                f"FAISS vectors: {index.ntotal}\n"
                f"Embedding rows: {len(embeddings)}\n"
                "Rebuild the FAISS index."
            )

        # --------------------------------------------------
        # DATASET / EMBEDDING COUNT CHECK
        # --------------------------------------------------

        if len(data) != len(
            embeddings
        ):

            raise RuntimeError(
                "Dataset / embedding alignment failure.\n"
                f"Dataset documents: {len(data)}\n"
                f"Embedding rows: {len(embeddings)}"
            )

        print(
            "[MS-ARC] FAISS consistency checks: PASS"
        )

        return index

    # ------------------------------------------------------
    # INDEX DOES NOT EXIST
    # ------------------------------------------------------

    print(
        "[MS-ARC] Existing FAISS index not found."
    )

    print(
        "[MS-ARC] Building HNSW FAISS Index..."
    )

    index = build_faiss(
        embeddings,
        index_type="hnsw",
    )

    FAISS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(FAISS_PATH),
    )

    print(
        "[MS-ARC] HNSW FAISS index saved."
    )

    return index


faiss_index = _load_or_build_faiss()


# ==========================================================
# FAISS REPORT
# ==========================================================

faiss_metadata = get_faiss_metadata(
    faiss_index
)

print()

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

print(
    f"[MS-ARC] FAISS index type: "
    f"{faiss_metadata.get('index_type')}"
)

print(
    f"[MS-ARC] FAISS metric: "
    f"{faiss_metadata.get('metric')}"
)


# ==========================================================
# BM25
# ==========================================================

print(
    "[MS-ARC] Initializing BM25..."
)

(
    bm25,
    tokenized,
    bm25_loaded_from_cache,
) = load_or_build_bm25(
    texts=texts,
    cache_path=BM25_CACHE_PATH,
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
# RETRIEVAL ENGINE READY
# ==========================================================

print(
    "[MS-ARC] Retrieval Engine Ready."
)

print(
    "[MS-ARC] Production FAISS index:",
    faiss_metadata.get(
        "index_type"
    ),
)


# ==========================================================
# RETRIEVE
# ==========================================================

def retrieve(
    state: RetrievalState,
    fusion_method: str = "minmax",
) -> RetrievalState:
    """
    Perform hybrid retrieval.

    FAISS returns row indices into `data`.

    Dataset `id` values are not used as FAISS row
    identifiers because the dataset contains duplicate
    IDs.

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
            "Unsupported fusion method: "
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

    # ======================================================
    # CANDIDATE DEPTH
    # ======================================================

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

        doc_id = item[
            "doc_id"
        ]

        if doc_id not in doc_lookup:

            raise RuntimeError(
                "FAISS returned an invalid "
                f"document row index: {doc_id}"
            )

        original = doc_lookup[
            doc_id
        ]

        dense_docs.append(

            RetrievedDocument(

                doc_id=str(
                    doc_id
                ),

                text=item[
                    "text"
                ],

                dense_score=float(
                    item[
                        "dense_score"
                    ]
                ),

                sparse_score=0.0,

                metadata={

                    "dataset_id":
                        original.get(
                            "id"
                        ),

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

                    "faiss_row":
                        doc_id,

                },
            )
        )

    # ======================================================
    # SPARSE RESULTS
    # ======================================================

    for item in results[
        "sparse_results"
    ]:

        doc_id = item[
            "doc_id"
        ]

        if doc_id not in doc_lookup:

            raise RuntimeError(
                "BM25 returned an invalid "
                f"document row index: {doc_id}"
            )

        original = doc_lookup[
            doc_id
        ]

        sparse_docs.append(

            RetrievedDocument(

                doc_id=str(
                    doc_id
                ),

                text=item[
                    "text"
                ],

                dense_score=0.0,

                sparse_score=float(
                    item[
                        "bm25_score"
                    ]
                ),

                metadata={

                    "dataset_id":
                        original.get(
                            "id"
                        ),

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

                    "faiss_row":
                        doc_id,

                },
            )
        )

    # ======================================================
    # HYBRID RESULTS
    # ======================================================

    for item in results[
        "merged_results"
    ]:

        doc_id = item[
            "doc_id"
        ]

        if doc_id not in doc_lookup:

            raise RuntimeError(
                "Hybrid retrieval returned "
                f"an invalid document row index: {doc_id}"
            )

        original = doc_lookup[
            doc_id
        ]

        merged_docs.append(

            RetrievedDocument(

                doc_id=str(
                    doc_id
                ),

                text=item[
                    "text"
                ],

                dense_score=float(
                    item[
                        "dense_score"
                    ]
                ),

                sparse_score=float(
                    item[
                        "bm25_score"
                    ]
                ),

                metadata={

                    "dataset_id":
                        original.get(
                            "id"
                        ),

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
                            item[
                                "hybrid_score"
                            ]
                        ),

                    "ranking_score":
                        float(
                            item.get(
                                "ranking_score",
                                item.get(
                                    "hybrid_score",
                                    0.0,
                                ),
                            )
                        ),

                    "lexical_anchor_score":
                        float(
                            item.get(
                                "lexical_anchor_score",
                                0.0,
                            )
                        ),

                    "exact_phrase_match":
                        bool(
                            item.get(
                                "exact_phrase_match",
                                False,
                            )
                        ),

                    "reference_match":
                        bool(
                            item.get(
                                "reference_match",
                                False,
                            )
                        ),

                    "matched_focus":
                        list(
                            item.get(
                                "matched_focus",
                                [],
                            )
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

                    "faiss_row":
                        doc_id,

                },
            )
        )

    # ======================================================
    # STORE STATE
    # ======================================================

    state.dense_results = (
        dense_docs
    )

    state.sparse_results = (
        sparse_docs
    )

    state.merged_results = (
        merged_docs
    )

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
        "faiss_index_type"
    ] = faiss_metadata.get(
        "index_type"
    )

    state.debug[
        "faiss_index_dimension"
    ] = faiss_metadata.get(
        "dimension"
    )

    state.debug[
        "faiss_index_vectors"
    ] = faiss_metadata.get(
        "ntotal"
    )

    state.debug[
        "bm25_cache_path"
    ] = str(
        BM25_CACHE_PATH
    )

    state.debug[
        "bm25_loaded_from_cache"
    ] = bm25_loaded_from_cache

    print(
        f"[MS-ARC] Retrieved "
        f"{len(merged_docs)} documents."
    )

    print(
        "[MS-ARC] Fusion method used:",
        fusion_method,
    )

    print(
        "[MS-ARC] FAISS index used:",
        faiss_metadata.get(
            "index_type"
        ),
    )

    print(
        "[MS-ARC] BM25 cache reused:",
        bm25_loaded_from_cache,
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
        "FAISS index:",
        state.debug.get(
            "faiss_index_type"
        ),
    )

    print(
        "FAISS vectors:",
        state.debug.get(
            "faiss_index_vectors"
        ),
    )

    print(
        "BM25 cache reused:",
        state.debug.get(
            "bm25_loaded_from_cache"
        ),
    )

    print()

    for doc in state.selected_documents:

        print(
            "FAISS row:",
            doc.doc_id,
            "| dataset id:",
            doc.metadata.get(
                "dataset_id"
            ),
            "| source:",
            doc.metadata.get(
                "source"
            ),
            "| hybrid:",
            doc.metadata.get(
                "hybrid_score"
            ),
        )