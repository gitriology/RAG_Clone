from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def validate_answer(answer, documents, threshold=0.4):

    if not documents:
        return {
            "is_valid": False,
            "confidence": 0.0
        }

    answer_embedding = model.encode([answer])
    doc_embeddings = model.encode([doc["text"] for doc in documents])

    scores = cosine_similarity(answer_embedding, doc_embeddings)[0]

    return {
        "is_valid": max(scores) >= threshold,
        "confidence": float(max(scores))
    }