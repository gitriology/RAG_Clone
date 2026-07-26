from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def validate_context(query, documents, threshold=0.30):
    """
    Computes context relevance.

    Documents are NOT discarded.
    They are simply given a context score.

    This allows later stages to decide whether the
    retrieved evidence is sufficient.
    """

    if not documents:
        return []

    query_embedding = model.encode([query])
    doc_embeddings = model.encode([doc["text"] for doc in documents])

    scores = cosine_similarity(query_embedding, doc_embeddings)[0]

    for doc, score in zip(documents, scores):
        doc["context_score"] = float(score)

    return documents