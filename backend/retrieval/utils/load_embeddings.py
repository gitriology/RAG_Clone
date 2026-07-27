import numpy as np
from pathlib import Path


CACHE = Path("backend/retrieval/cache/embeddings.npy")


def load_embeddings():
    """
    Load cached embeddings used to build the FAISS index.
    """

    if not CACHE.exists():
        raise FileNotFoundError(
            "Embeddings cache not found.\n"
            "Run save_embeddings.py first."
        )

    return np.load(CACHE)