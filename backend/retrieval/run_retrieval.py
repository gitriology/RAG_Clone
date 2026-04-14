import sys
import os
import time
import numpy as np
import pickle
import faiss

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

EMBEDDINGS_PATH = "embeddings_cache.npy"
FAISS_PATH      = "faiss_cache.index"
BM25_PATH       = "bm25_cache.pkl"

print("Step 1: starting script")

import sys
import os
import time
import numpy as np
import pickle
import faiss

print("Step 2: basic imports done")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Step 3: loading project modules...")

from utils.load_data import load_data
print("Step 4: load_data imported")

from dense.embedder import get_embeddings, model
print("Step 5: embedder imported ← if stuck here = model loading")

from index.faiss_index import build_faiss
print("Step 6: faiss imported")

from lexical.bm25 import build_bm25, search_bm25
print("Step 7: bm25 imported")

from hybrid.hybrid_search import hybrid_search
print("Step 8: all imports done")

data = load_data("/data/processed/all_domains.json")

texts = [item["text"] for item in data]

# Build systems
if os.path.exists(EMBEDDINGS_PATH):
    # cache found → load from disk (fast)
    print("Loading embeddings from cache...")
    t = time.time()
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"Loaded in {time.time()-t:.2f}s")
else:
    # no cache → build and save (slow, first time only)
    print("Building embeddings for first time...")
    t = time.time()
    embeddings = get_embeddings(texts)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Built and saved in {time.time()-t:.2f}s")
if os.path.exists(FAISS_PATH):
    # cache found → load from disk (fast)
    print("Loading FAISS from cache...")
    t = time.time()
    faiss_index = faiss.read_index(FAISS_PATH)
    print(f"Loaded in {time.time()-t:.2f}s")
else:
    # no cache → build and save (slow, first time only)
    print("Building FAISS for first time...")
    t = time.time()
    faiss_index = build_faiss(embeddings)
    faiss.write_index(faiss_index, FAISS_PATH)
    print(f"Built and saved in {time.time()-t:.2f}s")
if os.path.exists(BM25_PATH):
    # cache found → load from disk (fast)
    print("Loading BM25 from cache...")
    t = time.time()
    with open(BM25_PATH, "rb") as f:
        bm25, tokenized = pickle.load(f)
    print(f"Loaded in {time.time()-t:.2f}s")
else:
    # no cache → build and save (slow, first time only)
    print("Building BM25 for first time...")
    t = time.time()
    bm25, tokenized = build_bm25(texts)
    with open(BM25_PATH, "wb") as f:
        pickle.dump((bm25, tokenized), f)
    print(f"Built and saved in {time.time()-t:.2f}s")

# Test query
# change this to anything you want, always fast
query = "Explain PCA"

t = time.time()
results = hybrid_search(query, model, faiss_index, bm25, tokenized, texts)
print(f"Search: {time.time()-t:.2f}s")

for r in results:
    print(r[:200])
    print("-" * 50)
