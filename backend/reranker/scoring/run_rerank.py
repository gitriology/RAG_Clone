from backend.retrieval.run_retrieval import retrieve
from backend.reranker.model.reranker import rerank
from backend.reranker.utils.context_selector import select_top_k

def run_pipeline(query):
    # Step 1: Retrieve
    retrieved_docs = retrieve(query, top_k=5)

    # Step 2: Rerank
    ranked_docs = rerank(query, retrieved_docs)

    # Step 3: Select best
    top_docs = select_top_k(ranked_docs, k=3)

    return top_docs


if __name__ == "__main__":
    query = "What is vaccination?"

    results = run_pipeline(query)

    for doc in results:
        print("Score:", doc["rerank_score"])
        print(doc["text"][:200])
        print("----")