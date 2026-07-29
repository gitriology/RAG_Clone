import numpy as np

from backend.retrieval.utils.load_data import load_data
from backend.retrieval.dense.embedder import get_embeddings

print("Loading dataset...")

data = load_data("data/processed/all_domains.json")

texts = [d["text"] for d in data]

print("Generating embeddings...")

embeddings = get_embeddings(texts)

np.save(
    "backend/optimization/cache/embeddings.npy",
    embeddings
)

print("Embeddings Saved Successfully!")
print("Shape :", embeddings.shape)