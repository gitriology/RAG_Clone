from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")


def validate_context(query, documents, threshold=0.3):
    """
    Filters irrelevant documents using cosine similarity
    """

    query_embedding = model.encode([query])
    doc_embeddings = model.encode([doc["text"] for doc in documents])

    scores = cosine_similarity(query_embedding, doc_embeddings)[0]

    validated_docs = []

    for doc, score in zip(documents, scores):
        if score >= threshold:
            doc["context_score"] = float(score)
            validated_docs.append(doc)

    return validated_docs