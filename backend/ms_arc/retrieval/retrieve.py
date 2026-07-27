import os
import faiss

from backend.retrieval.utils.load_data import load_data
from backend.retrieval.dense.embedder import model
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

data = load_data("data/processed/all_domains.json")


texts = [doc["text"] for doc in data]

# Fast lookup by document id
doc_lookup = {

    idx: doc

    for idx, doc in enumerate(data)

}

# ==========================================================
# EMBEDDINGS
# ==========================================================

print("[MS-ARC] Loading Cached Embeddings...")

from backend.retrieval.utils.load_embeddings import load_embeddings

embeddings = load_embeddings()

# ==========================================================
# FAISS
# ==========================================================

FAISS_PATH = "backend/retrieval/index/faiss.index"

if os.path.exists(FAISS_PATH):

    print("[MS-ARC] Loading Existing FAISS Index...")

    faiss_index = faiss.read_index(FAISS_PATH)

else:

    print("[MS-ARC] Building FAISS Index...")

    faiss_index = build_faiss(embeddings)

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

print("[MS-ARC] Building BM25 Index...")

bm25, tokenized = build_bm25(texts)

print("[MS-ARC] Retrieval Engine Ready.")

# ==========================================================
# RETRIEVE
# ==========================================================


def retrieve(
    state: RetrievalState
) -> RetrievalState:
    """
    Pure retrieval.

    No adaptive retrieval.

    No confidence.

    No evidence state.

    No reranking.

    This function ONLY performs
    Hybrid Retrieval.

    MS-ARC modules will decide what
    happens next.
    """

    print("\n[MS-ARC] Hybrid Retrieval")

    results = hybrid_search(
        query=state.query,
        faiss_index=faiss_index,
        bm25=bm25,
        texts=texts,
        k=state.recommended_topk,
    )

    dense_docs = []

    sparse_docs = []

    merged_docs = []

    # =====================================================
    # Dense Results
    # =====================================================

    for item in results["dense_results"]:

        original = doc_lookup[item["doc_id"]]

        dense_docs.append(

            RetrievedDocument(

                doc_id=str(item["doc_id"]),

                text=item["text"],

                dense_score=item["dense_score"],

                sparse_score=0.0,

                metadata={

                    "domain": original.get("domain", "general"),

                    "source": original.get("source", "unknown"),

                    "hybrid_score": 0.0

                }

            )

        )

    # =====================================================
    # Sparse Results
    # =====================================================

    for item in results["sparse_results"]:

        original = doc_lookup[item["doc_id"]]

        sparse_docs.append(

            RetrievedDocument(

                doc_id=str(item["doc_id"]),

                text=item["text"],

                dense_score=0.0,

                sparse_score=item["bm25_score"],

                metadata={

                    "domain": original.get("domain", "general"),

                    "source": original.get("source", "unknown"),

                    "hybrid_score": 0.0

                }

            )

        )

    # =====================================================
    # Hybrid Results
    # =====================================================

    for item in results["merged_results"]:

        original = doc_lookup[item["doc_id"]]

        merged_docs.append(

            RetrievedDocument(

                doc_id=str(item["doc_id"]),

                text=item["text"],

                dense_score=item["dense_score"],

                sparse_score=item["bm25_score"],

                metadata={

                    "domain": original.get("domain", "general"),

                    "source": original.get("source", "unknown"),

                    "hybrid_score": item["hybrid_score"]

                }

            )

        )

    state.dense_results = dense_docs

    state.sparse_results = sparse_docs

    state.merged_results = merged_docs

    state.selected_documents = merged_docs

    print(

        f"[MS-ARC] Retrieved "

        f"{len(merged_docs)} documents."

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

    state = retrieve(state)

    print()
    print(f"[MS-ARC] Dense Candidates   : {len(state.dense_results)}")
    print(f"[MS-ARC] Sparse Candidates  : {len(state.sparse_results)}")
    print(f"[MS-ARC] Final Top-K        : {len(state.selected_documents)}")
    print()

    for doc in state.selected_documents:

        print("-" * 60)

        print(doc.doc_id)

        print(doc.dense_score)

        print(doc.sparse_score)

        print(doc.metadata["hybrid_score"])

        print(doc.metadata["domain"])

        print(doc.text[:150])