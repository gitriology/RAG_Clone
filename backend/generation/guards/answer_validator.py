from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def validate_answer(answer, documents, threshold=0.4):
    """
    Checks if answer is grounded in retrieved docs
    """

    answer_embedding = model.encode([answer])
    doc_embeddings = model.encode([doc["text"] for doc in documents])

    scores = cosine_similarity(answer_embedding, doc_embeddings)[0]
    max_score = max(scores)

    return {
        "is_valid": bool(max_score >= threshold),   # ✅ FIXED
        "confidence": float(max_score)
    }