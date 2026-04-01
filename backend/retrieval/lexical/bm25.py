from rank_bm25 import BM25Okapi

def build_bm25(texts):
    tokenized = [text.split() for text in texts]
    return BM25Okapi(tokenized), tokenized

def search_bm25(query, bm25, tokenized, k=5):
    scores = bm25.get_scores(query.split())
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]