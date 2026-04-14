from utils.load_data import load_data
from dense.embedder import get_embeddings, model
from index.faiss_index import build_faiss
from lexical.bm25 import build_bm25, search_bm25
from hybrid.hybrid_search import hybrid_search

data = load_data("/data/processed/all_domains.json")

texts = [item["text"] for item in data]

# Build systems
embeddings = get_embeddings(texts)
faiss_index = build_faiss(embeddings)
bm25, tokenized = build_bm25(texts)

# Test query
query = "What is immunization?"

results = hybrid_search(query, model, faiss_index, bm25, tokenized, texts)

for r in results:
    print(r[:200])
    print("-"*50)
