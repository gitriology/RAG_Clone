import os
import faiss

from backend.retrieval.utils.load_data import load_data
from backend.retrieval.dense.embedder import model
from backend.retrieval.index.faiss_index import build_faiss
from backend.retrieval.lexical.bm25 import build_bm25
from backend.retrieval.hybrid.hybrid_search import hybrid_search


# ==============================
# Optimization Modules
# ==============================

from backend.optimization.query_complexity import analyze_query
from backend.optimization.adaptive_k import choose_k
from backend.optimization.confidence import retrieval_confidence
from backend.optimization.evidence_state import compute_evidence_state
from backend.optimization.load_embeddings import load_embeddings
from backend.optimization.domain_router import detect_domain
# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading Dataset...")

data = load_data("data/processed/all_domains.json")

texts = [doc["text"] for doc in data]

# Fast lookup dictionary
text_to_doc = {
    doc["text"]: doc
    for doc in data
}

# ==========================================================
# LOAD EMBEDDINGS
# ==========================================================

print("Loading Cached Embeddings...")

embeddings = load_embeddings()

# ==========================================================
# LOAD / BUILD FAISS
# ==========================================================

FAISS_PATH = "backend/retrieval/index/faiss.index"

if os.path.exists(FAISS_PATH):

    print("Loading Existing FAISS Index...")

    faiss_index = faiss.read_index(FAISS_PATH)

else:

    print("Building FAISS Index...")

    faiss_index = build_faiss(embeddings)

    faiss.write_index(
        faiss_index,
        FAISS_PATH
    )

# ==========================================================
# BUILD BM25
# ==========================================================

print("Building BM25 Index...")

bm25, tokenized = build_bm25(texts)

print("\nRetrieval System Ready.")

# ==========================================================
# RETRIEVE FUNCTION
# ==========================================================

def retrieve(query):

    print("\n" + "=" * 70)

    print("Optimization Module Enabled")
    print("Adaptive Retrieval : ON")
    print("Evidence State : ON")
    print("Dynamic Top-K : ON")
    print("Confidence Evaluation : ON")

    print("\nUSER QUERY :", query)

    # -----------------------------------------
    # STEP 1 : Query Complexity
    # -----------------------------------------

    complexity = analyze_query(query)

    top_k = choose_k(complexity)

    print("\nQuery Complexity :", complexity)

    print("Initial Retrieval K :", top_k)

    MAX_K = 12

    OVERALL_THRESHOLD = 0.70

    while True:

        # -----------------------------------------
        # STEP 2 : Hybrid Retrieval
        # -----------------------------------------

        results = hybrid_search(
            query=query,
            model=model,
            faiss_index=faiss_index,
            bm25=bm25,
            tokenized=tokenized,
            texts=texts,
            k=top_k
        )

        retrieved_docs = [
            text_to_doc[r]
            for r in results
            if r in text_to_doc
        ]

        # -----------------------------------------
        # STEP 3 : Confidence
        # -----------------------------------------

        confidence = retrieval_confidence(
            query,
            retrieved_docs
        )

        # -----------------------------------------
        # STEP 4 : Evidence State
        # -----------------------------------------

        evidence = compute_evidence_state(
            retrieved_docs
        )

        # Increase confidence slightly as coverage improves
        confidence += evidence["coverage"] * 0.10

        if confidence > 1:
            confidence = 1

        print(f"\nRetrieval Confidence : {confidence:.3f}")

        print("\nEvidence State")

        print(f"Coverage        : {evidence['coverage']:.3f}")
        print(f"Richness        : {evidence['richness']:.3f}")
        print(f"Diversity       : {evidence['diversity']:.3f}")
        print(f"Evidence Score  : {evidence['evidence_score']:.3f}")

        # -----------------------------------------
        # STEP 5 : Overall Score
        # -----------------------------------------

        overall_score = (

            confidence * 0.6

            +

            evidence["evidence_score"] * 0.4

        )

        print(f"\nOverall Retrieval Score : {overall_score:.3f}")

        # -----------------------------------------
        # STEP 6 : Adaptive Retrieval
        # -----------------------------------------

        if overall_score >= OVERALL_THRESHOLD:

            print("\nEvidence Sufficient.")

            break

        if top_k >= MAX_K:

            print("\nMaximum Retrieval Depth Reached.")

            break

        print("\nLow Evidence Detected.")

        print("Increasing Retrieval Depth...")

        top_k += 2

    print("\nFinal Retrieval K :", top_k)

    print("\nFinal Evidence State")

    print(evidence)

    print("\nRetrieval Completed Successfully")

    print("=" * 70)

    return retrieved_docs


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    query = "What is vaccination?"

    docs = retrieve(query)

    print("\nRetrieved Documents")

    for i, doc in enumerate(docs, start=1):

        print("\n" + "-" * 70)

        print(f"Document {i}")

        print("Domain :", doc["domain"])

        print(doc["text"][:250])

        print("-" * 70)