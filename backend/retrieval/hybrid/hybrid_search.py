from backend.retrieval.lexical.bm25 import search_bm25
def hybrid_search(query, model, faiss_index, bm25, tokenized, texts, k=5):
    
    # Dense
    q_emb = model.encode([query])
    dense_ids = faiss_index.search(q_emb, k)[1][0]

    # BM25
    bm25_ids = search_bm25(query, bm25, tokenized, k)

    # Merge
    combined = list(set(dense_ids) | set(bm25_ids))

    return [texts[i] for i in combined[:k]]