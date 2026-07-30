from backend.retrieval.run_retrieval import retrieve
from backend.reranker.model.reranker import rerank
from backend.reranker.utils.context_selector import select_top_k

from backend.generation.validation.context_validator import validate_context
from backend.generation.guards.answer_validator import validate_answer
from backend.generation.answer_generator import generate_answer

from backend.reranker.scoring.confidence.confidence_score import (
    compute_confidence,
)


# ==========================================================
# PIPELINE
# ==========================================================

def run_pipeline(query):

    # ------------------------------------------------------
    # Step 1 : Retrieval
    # ------------------------------------------------------

    retrieval_result = retrieve(query)

    retrieved_docs = retrieval_result["documents"]

    print("Retrieved Docs:", len(retrieved_docs))

    retrieval_confidence = retrieval_result[
        "retrieval_confidence"
    ]

    evidence = retrieval_result["evidence"]

    # ------------------------------------------------------
    # Step 2 : Reranking
    # ------------------------------------------------------

    ranked_docs = rerank(
        query,
        retrieved_docs
    )

    print("Ranked Docs:", len(ranked_docs))

    # ------------------------------------------------------
    # Step 3 : Context Validation
    # ------------------------------------------------------

    validated_docs = validate_context(
        query,
        ranked_docs
    )

    print("Validated Docs:", len(validated_docs))

    # ------------------------------------------------------
    # Step 4 : Select Top Documents
    # ------------------------------------------------------

    top_docs = select_top_k(
        validated_docs,
        k=3
    )

    if not top_docs:
        return []

    print("Top Docs:", len(top_docs))

    # ------------------------------------------------------
    # Step 5 : Generate Answer
    # ------------------------------------------------------

    answer = generate_answer(
        query=query,
        documents=top_docs,
    )

    # ------------------------------------------------------
    # Step 6 : Validate Answer
    # ------------------------------------------------------

    validation = validate_answer(
        answer,
        top_docs
    )

    # ------------------------------------------------------
    # Step 7 : Confidence
    # ------------------------------------------------------

    confidence = compute_confidence(
        documents=top_docs,
        validation=validation,
        retrieval_confidence=retrieval_confidence,
    )

    print("Pipeline Confidence:", confidence)
    print("Retrieval Confidence:", retrieval_confidence)
    print("Validation:", validation)

    MIN_CONFIDENCE = 0.40

    if confidence < MIN_CONFIDENCE:

        print("Low overall confidence.")

        return []

    # ------------------------------------------------------
    # Step 8 : Format Output
    # ------------------------------------------------------

    formatted_results = []

    for doc in top_docs:

        formatted_results.append(

            {

                "text": answer,

                "rerank_score": float(
                    doc.get("rerank_score", 0.0)
                ),

                "context_score": float(
                    doc.get("context_score", 0.0)
                ),

                "pipeline_confidence": float(
                    confidence
                ),

                "retrieval_confidence": float(
                    retrieval_confidence
                ),

                "answer_confidence": float(
                    validation["confidence"]
                ),

                "answer_valid": validation[
                    "is_valid"
                ],

                "domain": doc.get(
                    "domain",
                    "general",
                ),

                "source": doc.get(
                    "source",
                    "unknown",
                ),

                "evidence": evidence,

            }

        )

    return formatted_results


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    query = (
        "Where did Chandrayaan-3 land on the Moon "
        "and which organization developed the mission?"
    )

    result = run_pipeline(query)

    print("\nRESULT:\n")

    for i, doc in enumerate(result):

        print(f"\n--- Document {i+1} ---")

        print("Answer:", doc["text"])

        print("Pipeline Confidence:",
              doc["pipeline_confidence"])

        print("Domain:", doc["domain"])

        print("Source:", doc["source"])