import numpy as np

from backend.models.model_registry import ModelRegistry


def get_embeddings(texts):
    """
    Generate normalized document embeddings.
    """

    model = ModelRegistry.get_embedding_model()

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

    model = ModelRegistry.get_embedding_model()

    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    return np.asarray(
        embedding,
        dtype=np.float32
    )