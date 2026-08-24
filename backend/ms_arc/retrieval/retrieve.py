import os
import faiss

from backend.retrieval.utils.load_data import load_data
from backend.retrieval.index.faiss_index import build_faiss
from backend.retrieval.lexical.bm25 import build_bm25
from backend.retrieval.hybrid.hybrid_search import hybrid_search

from backend.ms_arc.state.retrieval_state import (
    RetrievalState,
    RetrievedDocument,
)

# ==========================================================
# DATASET
# ==========================================================

print("[MS-ARC] Loading Dataset...")

data = load_data(
    "data/processed/all_domains.json"
)

texts = [
    doc["text"]
    for doc in data
]

# Fast lookup by document id
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

# ==========================================================
# FAISS
# ==========================================================

FAISS_PATH = (
    "backend/retrieval/index/faiss.index"
)

if os.path.exists(FAISS_PATH):

    print(
        "[MS-ARC] "
        "Loading Existing FAISS Index..."
    )

    faiss_index = faiss.read_index(
        FAISS_PATH
    )

else:

    print(
        "[MS-ARC] "
        "Building FAISS Index..."
    )

    faiss_index = build_faiss(
        embeddings
    )

    faiss.write_index(
        faiss_index,
        FAISS_PATH
    )

print(len(data))
print(len(texts))
print(faiss_index.ntotal)

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
    fusion_method: str = "minmax",
) -> RetrievalState:
    """
    Performs hybrid retrieval.

    fusion_method is explicitly propagated through
    the MS-ARC pipeline.

    Supported:

        minmax
        rrf

    This explicit parameter is intentionally used
    instead of an environment variable so that
    benchmarking cannot accidentally reuse the
    previously imported fusion configuration.
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

    # ======================================================
    # Candidate depth
    # ======================================================
    #
    # Optimization #6 candidate controller may already
    # have selected a candidate depth.
    #
    # If present, use it.
    # Otherwise let hybrid_search use its default.
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
    # Dense Results
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
    # Sparse Results
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
    # Hybrid Results
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

    print(
        f"[MS-ARC] Retrieved "
        f"{len(merged_docs)} documents."
    )

    print(
        "[MS-ARC] Fusion method used:",
        fusion_method,
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