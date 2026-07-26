import os
import numpy as np
import faiss

from backend.retrieval.utils.load_data import load_data
from backend.retrieval.dense.embedder import get_embeddings

print("Loading Dataset...")

data = load_data("data/processed/all_domains.json")

texts = [d["text"] for d in data]

print("Generating Embeddings...")

embeddings = get_embeddings(texts)

os.makedirs("backend/retrieval/cache", exist_ok=True)

np.save(
    "backend/retrieval/cache/embeddings.npy",
    embeddings
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

faiss.write_index(
    index,
    "backend/retrieval/cache/faiss.index"
)

print("Index Saved Successfully")