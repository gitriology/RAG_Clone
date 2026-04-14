from backend.retrieval.utils.load_data import load_data
from backend.retrieval.dense.embedder import get_embeddings, model
from backend.retrieval.index.faiss_index import build_faiss
from backend.retrieval.lexical.bm25 import build_bm25
from backend.retrieval.hybrid.hybrid_search import hybrid_search

# ================================
# LOAD DATA (ONCE)
# ================================
data = load_data("data/processed/all_domains.json")

# Extract texts
texts = [item["text"] for item in data]

# 🔥 OPTIMIZATION: Fast lookup dictionary
text_to_doc = {item["text"]: item for item in data}

# ================================
# BUILD RETRIEVAL SYSTEMS (ONCE)
# ================================
embeddings = get_embeddings(texts)
faiss_index = build_faiss(embeddings)
bm25, tokenized = build_bm25(texts)


# ================================
# MAIN RETRIEVE FUNCTION
# ================================
def retrieve(query, top_k=5):
    """
    Hybrid retrieval:
    - Dense (FAISS)
    - Lexical (BM25)
    - Returns structured documents (not just text)
    """

    results = hybrid_search(
        query,
        model,
        faiss_index,
        bm25,
        tokenized,
        texts
    )

    # 🔥 FAST + CLEAN mapping (O(1) lookup)
    final_results = [
        text_to_doc[res]
        for res in results[:top_k]
        if res in text_to_doc
    ]

    return final_results


# ================================
# TEST RUN
# ================================
if __name__ == "__main__":
    query = "What is immunization?"

    results = retrieve(query)

    for r in results:
        print("Domain:", r["domain"])
        print("Score Input Text:", r["text"][:200])
        print("-" * 50)