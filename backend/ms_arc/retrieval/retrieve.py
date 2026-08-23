import os

import faiss


from backend.retrieval.utils.load_data import (
    load_data
)

from backend.retrieval.index.faiss_index import (
    build_faiss
)

from backend.retrieval.lexical.bm25 import (
    build_bm25
)

from backend.retrieval.hybrid.hybrid_search import (
    hybrid_search
)


from backend.ms_arc.state.retrieval_state import (
    RetrievalState,
    RetrievedDocument,
)


# ==========================================================
# DATASET
# ==========================================================

print(
    "[MS-ARC] Loading Dataset..."
)

data = load_data(
    "data/processed/all_domains.json"
)

texts = [
    doc["text"]
    for doc in data
]


# ==========================================================
# DOCUMENT LOOKUP
# ==========================================================

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
    load_embeddings
)

embeddings = load_embeddings()


# ==========================================================
# FAISS
# ==========================================================

FAISS_PATH = (
    "backend/retrieval/index/faiss.index"
)


if os.path.exists(
    FAISS_PATH
):

    print(
        "[MS-ARC] Loading Existing FAISS Index..."
    )

    faiss_index = faiss.read_index(
        FAISS_PATH
    )

else:

    print(
        "[MS-ARC] Building FAISS Index..."
    )

    faiss_index = build_faiss(
        embeddings
    )

    faiss.write_index(
        faiss_index,
        FAISS_PATH
    )


print(
    len(data)
)

print(
    len(texts)
)

print(
    faiss_index.ntotal
)


# ==========================================================
# BM25
# ==========================================================

print(
    "[MS-ARC] Building BM25 Index..."
)

bm25, tokenized = build_bm25(
    texts
)

print(
    "[MS-ARC] Retrieval Engine Ready."
)


# ==========================================================
# RETRIEVE
# ==========================================================

def retrieve(
    state: RetrievalState,
    candidate_k=None,
    fusion_method="minmax",
) -> RetrievalState:

    """
    Performs hybrid retrieval.

    Optimization #6
    ----------------
    candidate_k is supplied by the MS-ARC adaptive
    candidate controller.

    Optimization #5
    ----------------
    Stability retrieval is handled separately.

    Optimization #4
    ----------------
    CrossEncoder reranking is not performed here.
    """

    print(
        "\n[MS-ARC] Hybrid Retrieval"
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

    # ======================================================
    # Dense Results
    # ======================================================

    dense_docs = []

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

                dense_score=item[
                    "dense_score"
                ],

                sparse_score=0.0,

                metadata={

                    "domain":
                        original.get(
                            "domain",
                            "general"
                        ),

                    "source":
                        original.get(
                            "source",
                            "unknown"
                        ),

                    "hybrid_score":
                        0.0,

                    "dense_raw_score":
                        item.get(
                            "dense_raw_score",
                            0.0
                        ),

                    "fusion_method":
                        results[
                            "fusion_method"
                        ],

                },

            )

        )

    # ======================================================
    # Sparse Results
    # ======================================================

    sparse_docs = []

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

                sparse_score=item[
                    "bm25_score"
                ],

                metadata={

                    "domain":
                        original.get(
                            "domain",
                            "general"
                        ),

                    "source":
                        original.get(
                            "source",
                            "unknown"
                        ),

                    "hybrid_score":
                        0.0,

                    "bm25_raw_score":
                        item.get(
                            "bm25_raw_score",
                            0.0
                        ),

                    "fusion_method":
                        results[
                            "fusion_method"
                        ],

                },

            )

        )

    # ======================================================
    # Hybrid Results
    # ======================================================

    merged_docs = []

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

                dense_score=item[
                    "dense_score"
                ],

                sparse_score=item[
                    "bm25_score"
                ],

                metadata={

                    "domain":
                        original.get(
                            "domain",
                            "general"
                        ),

                    "source":
                        original.get(
                            "source",
                            "unknown"
                        ),

                    "hybrid_score":
                        item[
                            "hybrid_score"
                        ],

                    "dense_raw_score":
                        item.get(
                            "dense_raw_score",
                            0.0
                        ),

                    "bm25_raw_score":
                        item.get(
                            "bm25_raw_score",
                            0.0
                        ),

                    "dense_rank":
                        item.get(
                            "dense_rank"
                        ),

                    "sparse_rank":
                        item.get(
                            "sparse_rank"
                        ),

                    "fusion_method":
                        results[
                            "fusion_method"
                        ],

                },

            )

        )

    # ======================================================
    # STORE RESULTS
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
    # OPTIMIZATION #6 DEBUG
    # ======================================================

    state.debug[
        "candidate_pool_used"
    ] = results[
        "candidate_k"
    ]

    state.debug[
        "fusion_method"
    ] = results[
        "fusion_method"
    ]

    state.debug[
        "dense_weight"
    ] = results[
        "dense_weight"
    ]

    state.debug[
        "sparse_weight"
    ] = results[
        "sparse_weight"
    ]

    state.debug[
        "rrf_k"
    ] = results[
        "rrf_k"
    ]

    state.debug[
        "retrieved_document_count"
    ] = len(
        merged_docs
    )

    print(
        "[MS-ARC] Retrieved "
        f"{len(merged_docs)} documents."
    )

    print(
        "[Optimization #6] "
        f"Candidate pool used: "
        f"{results['candidate_k']}"
    )

    print(
        "[Optimization #6] "
        f"Fusion method: "
        f"{results['fusion_method']}"
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
        candidate_k=20,
        fusion_method="minmax",
    )

    print()

    print(
        "[MS-ARC] Dense Candidates   : "
        f"{len(state.dense_results)}"
    )

    print(
        "[MS-ARC] Sparse Candidates  : "
        f"{len(state.sparse_results)}"
    )

    print(
        "[MS-ARC] Final Top-K        : "
        f"{len(state.selected_documents)}"
    )

    print(
        "[MS-ARC] Fusion             : "
        f"{state.debug.get('fusion_method')}"
    )

    print()

    for doc in state.selected_documents:

        print(
            "-" * 60
        )

        print(
            doc.doc_id
        )

        print(
            doc.dense_score
        )

        print(
            doc.sparse_score
        )

        print(
            doc.metadata[
                "hybrid_score"
            ]
        )

        print(
            doc.metadata[
                "domain"
            ]
        )

        print(
            doc.text[:150]
        )