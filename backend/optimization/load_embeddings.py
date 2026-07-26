import numpy as np
from pathlib import Path

def load_embeddings():
    CACHE = Path("backend/optimization/cache/embeddings.npy")
    if not CACHE.exists():
        raise FileNotFoundError(
            "Embeddings cache not found. Run "
            "'python -m backend.optimization.save_embeddings' first."
        )

    return np.load(CACHE)