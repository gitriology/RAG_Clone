from sentence_transformers import SentenceTransformer
import numpy as np

# ==========================================
# Embedding Model
# ==========================================

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)


def get_embeddings(texts):
    """
    Generate normalized document embeddings.
    """

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return np.asarray(
        embeddings,
        dtype=np.float32
    )


def encode_query(query):
    """
    Generate normalized query embedding.
    """

    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    return np.asarray(
        embedding,
        dtype=np.float32
    )