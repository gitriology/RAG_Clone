from sklearn.metrics.pairwise import cosine_similarity

from backend.models.model_registry import ModelRegistry


def validate_answer(answer, documents, threshold=0.4):

    if not documents:
        return {
            "is_valid": False,
            "confidence": 0.0
        }

    model = ModelRegistry.get_validation_model()

    answer_embedding = model.encode([answer])
    doc_embeddings = model.encode([doc["text"] for doc in documents])

    scores = cosine_similarity(answer_embedding, doc_embeddings)[0]

    print(
    "[Answer Validation] Model ID:",
    id(model)
)

    return {
        "is_valid": max(scores) >= threshold,
        "confidence": float(max(scores))
    }