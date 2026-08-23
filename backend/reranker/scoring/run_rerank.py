from backend.retrieval.run_retrieval import (
    retrieve
)

from backend.reranker.model.reranker import (
    rerank
)

from backend.reranker.utils.context_selector import (
    select_top_k
)

from backend.generation.validation.context_validator import (
    validate_context
)

from backend.generation.guards.answer_validator import (
    validate_answer
)

from backend.generation.answer_generator import (
    generate_answer
)

from backend.generation.evidence_fusion import (
    EvidenceFusion
)

from backend.reranker.scoring.confidence.confidence_score import (
    compute_confidence
)


# ==========================================================
# PIPELINE
# ==========================================================

def run_pipeline(query):

    # ======================================================
    # STEP 1 : RETRIEVAL + MS-ARC
    # ======================================================

    retrieval_result = retrieve(
        query
    )

    retrieved_docs = (
        retrieval_result[
            "documents"
        ]
    )

    print(
        "Retrieved Docs:",
        len(retrieved_docs)
    )

    retrieval_confidence = (
        retrieval_result[
            "retrieval_confidence"
        ]
    )

    evidence = (
        retrieval_result[
            "evidence"
        ]
    )

    # ======================================================
    # VERIFY OPTIMIZATION #4
    # ======================================================

    missing_scores = [

        doc

        for doc in retrieved_docs

        if "rerank_score" not in doc

    ]

    if missing_scores:

        raise RuntimeError(

            "Optimization #4 failed: "

            "MS-ARC did not propagate "

            "rerank_score to downstream documents."

        )

    # ======================================================
    # OPTIMIZATION #6 DEBUG
    # ======================================================

    print(
        "[Optimization #6] "
        f"Initial candidate K: "
        f"{evidence.get('initial_candidate_k')}"
    )

    print(
        "[Optimization #6] "
        f"Adaptive candidate K: "
        f"{evidence.get('adaptive_candidate_k')}"
    )

    print(
        "[Optimization #6] "
        f"Final candidate K: "
        f"{evidence.get('final_candidate_k')}"
    )

    print(
        "[Optimization #6] "
        f"Candidate expanded: "
        f"{evidence.get('candidate_expanded')}"
    )

    print(
        "[Optimization #6] "
        f"Fusion method: "
        f"{evidence.get('fusion_method')}"
    )

    # ======================================================
    # STEP 2 : REUSE MS-ARC CROSSENCODER RESULTS
    # ======================================================

    ranked_docs = rerank(
        query,
        retrieved_docs,
    )

    print(
        "Ranked Docs:",
        len(ranked_docs)
    )

    # ======================================================
    # STEP 3 : CONTEXT VALIDATION
    # ======================================================

    validated_docs = validate_context(
        query,
        ranked_docs,
    )

    print(
        "Validated Docs:",
        len(validated_docs)
    )

    # ======================================================
    # STEP 4 : TOP DOCUMENTS
    # ======================================================

    top_docs = select_top_k(
        validated_docs,
        k=3,
    )

    if not top_docs:

        return []

    print(
        "Top Docs:",
        len(top_docs)
    )

    # ======================================================
    # STEP 5 : GENERATION
    # ======================================================

    generation = generate_answer(

        query=query,

        documents=top_docs,

    )

    # ======================================================
    # STEP 5.1 : EVIDENCE FUSION
    # ======================================================

    fusion = EvidenceFusion()

    fusion_result = fusion.fuse(
        generation
    )

    paragraph = (
        fusion_result[
            "paragraph"
        ]
    )

    answer = paragraph

    # ======================================================
    # STEP 6 : ANSWER VALIDATION
    # ======================================================

    validation = validate_answer(

        answer,

        top_docs,

    )

    # ======================================================
    # STEP 7 : CONFIDENCE
    # ======================================================

    confidence = compute_confidence(

        documents=top_docs,

        validation=validation,

        retrieval_confidence=
            retrieval_confidence,

    )

    print(
        "Pipeline Confidence:",
        confidence
    )

    print(
        "Retrieval Confidence:",
        retrieval_confidence
    )

    print(
        "Validation:",
        validation
    )

    # ======================================================
    # MINIMUM CONFIDENCE
    # ======================================================

    MIN_CONFIDENCE = 0.40

    if confidence < MIN_CONFIDENCE:

        print(
            "Low overall confidence."
        )

        return []

    # ======================================================
    # FINAL OUTPUT
    # ======================================================

    formatted_results = []

    for doc in top_docs:

        formatted_results.append({

            "text":
                answer,

            "rerank_score":
                float(
                    doc.get(
                        "rerank_score",
                        0.0
                    )
                ),

            "context_score":
                float(
                    doc.get(
                        "context_score",
                        0.0
                    )
                ),

            "pipeline_confidence":
                float(
                    confidence
                ),

            "retrieval_confidence":
                float(
                    retrieval_confidence
                ),

            "answer_confidence":
                float(
                    validation[
                        "confidence"
                    ]
                ),

            "answer_valid":
                validation[
                    "is_valid"
                ],

            "domain":
                doc.get(
                    "domain",
                    "general"
                ),

            "source":
                doc.get(
                    "source",
                    "unknown"
                ),

            "evidence":
                evidence,

        })

    return formatted_results


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    query = (

        "Where did Chandrayaan-3 land on the Moon "

        "and which organization developed the mission?"

    )

    result = run_pipeline(
        query
    )

    print(
        "\nRESULT:\n"
    )

    for i, doc in enumerate(
        result
    ):

        print(
            f"\n--- Document {i + 1} ---"
        )

        print(
            "Answer:",
            doc["text"]
        )

        print(
            "Rerank Score:",
            doc[
                "rerank_score"
            ]
        )

        print(
            "Pipeline Confidence:",
            doc[
                "pipeline_confidence"
            ]
        )

        print(
            "Domain:",
            doc[
                "domain"
            ]
        )

        print(
            "Source:",
            doc[
                "source"
            ]
        )

        print(
            "Optimization #6 Evidence:",
            doc[
                "evidence"
            ]
        )