from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieval_confidence(query, retrieved_docs):

    query_embedding = model.encode([query])

    doc_embeddings = model.encode(
        [doc["text"] for doc in retrieved_docs]
    )

    scores = cosine_similarity(
        query_embedding,
        doc_embeddings
    )[0]

    return max(scores)