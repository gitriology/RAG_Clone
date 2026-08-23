import time

from sentence_transformers import util

from backend.models.model_registry import ModelRegistry


def validate_context(query, documents, threshold=0.30):
    """
    Computes context relevance.

    Documents are NOT discarded.
    They are simply given a context score.

    This allows later stages to decide whether the
    retrieved evidence is sufficient.

    Optimization #2:
    - Uses the shared MiniLM validation model
      from ModelRegistry.
    - Encodes the query once.
    - Encodes all documents in one batch.
    - Uses normalized embeddings.
    - Computes cosine similarity in a vectorized operation.
    """

    # ------------------------------------------------------
    # Empty input
    # ------------------------------------------------------

    if not documents:
        return []

    # ------------------------------------------------------
    # Get shared validation model
    # ------------------------------------------------------

    model = ModelRegistry.get_validation_model()

    # ------------------------------------------------------
    # Collect document text
    # ------------------------------------------------------

    document_texts = [
        doc["text"]
        for doc in documents
    ]

    # ------------------------------------------------------
    # Start timing
    # ------------------------------------------------------

    start_time = time.perf_counter()

    # ------------------------------------------------------
    # Encode query ONCE
    # ------------------------------------------------------

    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    # ------------------------------------------------------
    # Encode ALL documents in ONE batch
    # ------------------------------------------------------

    doc_embeddings = model.encode(
        document_texts,
        batch_size=32,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    # ------------------------------------------------------
    # Vectorized cosine similarity
    # ------------------------------------------------------

    scores = util.cos_sim(
        query_embedding,
        doc_embeddings,
    )[0]

    # ------------------------------------------------------
    # Attach context scores
    # ------------------------------------------------------

    for doc, score in zip(
        documents,
        scores,
    ):

        doc["context_score"] = float(score)

    # ------------------------------------------------------
    # Timing / debug
    # ------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        f"[Context Validation] "
        f"Documents: {len(documents)}"
    )

    print(
        f"[Context Validation] "
        f"Embedding + similarity time: "
        f"{elapsed:.4f}s"
    )

    print(
    "[Context Validation] Model ID:",
    id(model)
)

    # ------------------------------------------------------
    # Documents are NOT filtered
    # ------------------------------------------------------

    return documents