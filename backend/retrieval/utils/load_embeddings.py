from __future__ import annotations

from pathlib import Path

import numpy as np


# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]


CACHE = (
    PROJECT_ROOT
    / "backend"
    / "retrieval"
    / "cache"
    / "embeddings.npy"
)


# ==========================================================
# LOAD
# ==========================================================

def load_embeddings():

    """
    Load cached document embeddings.

    The path is resolved relative to the project root,
    rather than the current working directory.
    """

    if not CACHE.exists():

        raise FileNotFoundError(

            "Embeddings cache not found.\n"

            f"Expected location:\n"
            f"{CACHE}\n\n"

            "Run the embedding generation/cache step first."
        )

    embeddings = np.load(
        CACHE
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:

        raise ValueError(

            "Invalid embeddings cache. "

            "Expected a 2D array with shape "

            "(num_documents, embedding_dimension). "
            f"Got shape: {embeddings.shape}"
        )

    if embeddings.shape[0] == 0:

        raise ValueError(
            "Embeddings cache contains zero vectors."
        )

    print(
        f"[Embeddings] Loaded: "
        f"{embeddings.shape}"
    )

    return embeddings