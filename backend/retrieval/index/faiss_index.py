import faiss
import numpy as np


def build_faiss(embeddings):
    """
    Build a FAISS Inner Product index.

    NOTE:
    Embeddings must already be L2-normalized.
    BGE embeddings are normalized in embedder.py.
    """

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    dimension = embeddings.shape[1]

    print(f"[FAISS] Embedding Dimension : {dimension}")

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(f"[FAISS] Indexed {index.ntotal} vectors")

    return index


def search_faiss(
    query_embedding,
    index,
    k=5
):
    """
    Search the FAISS index.

    Returns
    -------
    scores : cosine similarities
    indices : document ids
    """

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    scores, indices = index.search(
        query_embedding,
        k
    )

    return scores[0], indices[0]