import numpy as np
from pathlib import Path

from backend.retrieval.utils.load_data import load_data
from backend.retrieval.dense.embedder import get_embeddings


def main():

    print("[Embedding] Loading dataset...")

    data = load_data(
        "data/processed/all_domains.json"
    )

    texts = [
        doc["text"]
        for doc in data
    ]

    print(f"[Embedding] Documents : {len(texts)}")

    print("[Embedding] Generating BGE embeddings...")

    embeddings = get_embeddings(texts)

    cache_dir = Path(
        "backend/retrieval/cache"
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        cache_dir / "embeddings.npy",
        embeddings
    )

    print()

    print("[Embedding] Saved Successfully")

    print(
        "Shape:",
        embeddings.shape
    )

    print(
        "Dtype:",
        embeddings.dtype
    )


if __name__ == "__main__":
    main()