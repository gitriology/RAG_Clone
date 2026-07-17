from backend.retrieval.run_retrieval import retrieve
from backend.reranker.model.reranker import rerank
from backend.reranker.utils.context_selector import select_top_k
from backend.generation.validation.context_validator import validate_context
from backend.generation.guards.answer_validator import validate_answer


def run_pipeline(query):
    # Step 1: Retrieval
    retrieved_docs = retrieve(query)

    # Step 2: Reranking
    ranked_docs = rerank(query, retrieved_docs)

    # Step 3: Context validation
    validated_docs = validate_context(query, ranked_docs)

    # Step 4: Select top docs
    top_docs = select_top_k(validated_docs, k=3)

    # Step 5: Simulated answer (LLM placeholder)
    answer = " ".join([
    doc["text"].split(".")[0] for doc in top_docs
])

    # Step 6: Answer validation
    validation = validate_answer(answer, top_docs)

    return {
        "answer": answer,
        "documents": top_docs,
        "validation": validation
    }


if __name__ == "__main__":
    query = "What is vaccination?"

    result = run_pipeline(query)

    print("\nANSWER:\n", result["answer"])
    print("\nVALIDATION:", result["validation"])