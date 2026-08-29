"""
Production Reranking / Answer Pipeline

Flow
----
Hybrid Retrieval
        ↓
MS-ARC reranking
        ↓
Context Validation
        ↓
Top-K Context
        ↓
Question-Aware Evidence Selection
        ↓
Evidence Fusion
        ↓
Answer Validation
        ↓
Confidence
        ↓
Final Result

Optimization #4
---------------
Reuse CrossEncoder scores produced by MS-ARC.

Optimization #17
----------------
Reuse sentence embeddings between Phase 10.1
and Phase 10.2.
"""

from __future__ import annotations

from typing import Any, Dict, List


from backend.retrieval.run_retrieval import (
    retrieve,
)

from backend.reranker.model.reranker import (
    rerank,
)

from backend.reranker.utils.context_selector import (
    select_top_k,
)

from backend.generation.validation.context_validator import (
    validate_context,
)

from backend.generation.guards.answer_validator import (
    validate_answer,
)

from backend.generation.answer_generator import (
    generate_answer,
)

from backend.generation.evidence_fusion import (
    EvidenceFusion,
)

from backend.reranker.scoring.confidence.confidence_score import (
    compute_confidence,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

TOP_K_DOCUMENTS = 3

MIN_CONFIDENCE = 0.40

DEFAULT_QUERY = (
    "Where did Chandrayaan-3 land on the Moon "
    "and which organization developed the mission?"
)


# ==========================================================
# SAFE FLOAT
# ==========================================================

def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def run_pipeline(
    query: str,
) -> List[Dict]:

    query = (
        query.strip()
        if query
        else DEFAULT_QUERY
    )

    # ======================================================
    # STEP 1
    # RETRIEVAL + MS-ARC
    # ======================================================

    retrieval_result = retrieve(
        query
    )

    retrieved_docs = retrieval_result.get(
        "documents",
        [],
    )

    retrieval_confidence = safe_float(
        retrieval_result.get(
            "retrieval_confidence",
            0.0,
        )
    )

    evidence = retrieval_result.get(
        "evidence",
        {},
    )

    print()
    print(
        "[MS-ARC] Retrieved Docs:",
        len(retrieved_docs),
    )

    print(
        "[MS-ARC] Retrieval Confidence:",
        retrieval_confidence,
    )

    # ======================================================
    # OPTIMIZATION #6 DEBUG
    # ======================================================

    if isinstance(
        evidence,
        dict,
    ):

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
    # OPTIMIZATION #4
    # Verify MS-ARC scores
    # ======================================================

    if retrieved_docs:

        missing_scores = [
            doc
            for doc in retrieved_docs
            if "rerank_score" not in doc
        ]

        if missing_scores:

            raise RuntimeError(
                "Optimization #4 failed: "
                "MS-ARC did not propagate "
                "rerank_score to downstream "
                "documents."
            )

        print(
            "[Optimization #4] "
            "MS-ARC rerank scores available."
        )

    # ======================================================
    # STEP 2
    # RERANK
    # ======================================================

    ranked_docs = rerank(
        query,
        retrieved_docs,
    )

    print(
        "Ranked Docs:",
        len(ranked_docs),
    )

    # ======================================================
    # STEP 3
    # CONTEXT VALIDATION
    # ======================================================

    validated_docs = validate_context(
        query,
        ranked_docs,
    )

    print(
        "Validated Docs:",
        len(validated_docs),
    )

    # ======================================================
    # STEP 4
    # TOP DOCUMENTS
    # ======================================================

    top_docs = select_top_k(
        validated_docs,
        k=TOP_K_DOCUMENTS,
    )

    if not top_docs:

        print(
            "No usable documents."
        )

        return []

    print(
        "Top Docs:",
        len(top_docs),
    )

    # ======================================================
    # STEP 5
    # QUESTION-AWARE GENERATION
    # ======================================================

    generation = generate_answer(
        query=query,
        documents=top_docs,
        max_sentences=3,
    )

    # ======================================================
    # OPTIMIZATION #17
    # REUSABLE SENTENCE EMBEDDINGS
    # ======================================================

    sentence_embeddings = generation.get(
        "sentence_embeddings",
        {},
    )

    print()
    print(
        "[Optimization #17] "
        f"Reusable sentence embeddings: "
        f"{len(sentence_embeddings)}"
    )

    if generation.get(
        "embedding_reuse_enabled",
        False,
    ):

        print(
            "[Optimization #17] "
            "Sentence embedding reuse: ENABLED"
        )

    else:

        print(
            "[Optimization #17] "
            "Sentence embedding reuse: DISABLED"
        )

    # ======================================================
    # STEP 5.1
    # EVIDENCE FUSION
    # ======================================================

    fusion = EvidenceFusion(
        sentence_embeddings=sentence_embeddings,
    )

    fusion_result = fusion.fuse(
        generation
    )

    answer = fusion_result.get(
        "paragraph",
        "",
    ).strip()

    # Fallback to Phase 10.1 if fusion somehow
    # produces no paragraph.
    if not answer:

        answer = generation.get(
            "answer",
            "",
        ).strip()

    # ======================================================
    # OPTIMIZATION #17 DEBUG
    # ======================================================

    embedding_reuses = int(
        fusion_result.get(
            "embedding_reuses",
            0,
        )
    )

    embedding_fallbacks = int(
        fusion_result.get(
            "embedding_fallbacks",
            0,
        )
    )

    print()
    print(
        "[Optimization #17] "
        f"Embedding reuses: "
        f"{embedding_reuses}"
    )

    print(
        "[Optimization #17] "
        f"Embedding fallbacks: "
        f"{embedding_fallbacks}"
    )

    # ======================================================
    # STEP 6
    # ANSWER VALIDATION
    # ======================================================

    validation = validate_answer(
        answer,
        top_docs,
    )

    answer_confidence = safe_float(
        validation.get(
            "confidence",
            0.0,
        )
    )

    answer_valid = bool(
        validation.get(
            "is_valid",
            False,
        )
    )

    # ======================================================
    # STEP 7
    # PIPELINE CONFIDENCE
    # ======================================================

    confidence = safe_float(
        compute_confidence(
            documents=top_docs,
            validation=validation,
            retrieval_confidence=(
                retrieval_confidence
            ),
        )
    )

    print()
    print(
        "Pipeline Confidence:",
        confidence,
    )

    print(
        "Retrieval Confidence:",
        retrieval_confidence,
    )

    print(
        "Answer Confidence:",
        answer_confidence,
    )

    print(
        "Answer Valid:",
        answer_valid,
    )

    print()
    print(
        "Question Targets:",
        generation.get(
            "question_targets",
            [],
        ),
    )

    # ======================================================
    # MINIMUM CONFIDENCE
    # ======================================================

    if confidence < MIN_CONFIDENCE:

        print(
            "Low overall confidence."
        )

        return []

    # ======================================================
    # SOURCE DOCUMENTS
    # ======================================================

    source_documents = []

    seen_sources = set()

    for doc in top_docs:

        source = doc.get(
            "source",
            "unknown",
        )

        doc_id = doc.get(
            "doc_id"
        )

        key = (
            doc_id,
            source,
        )

        if key in seen_sources:
            continue

        seen_sources.add(
            key
        )

        source_documents.append(
            {
                "doc_id": doc_id,
                "source": source,
                "domain": doc.get(
                    "domain",
                    "general",
                ),
            }
        )

    # ======================================================
    # SELECTED EVIDENCE
    # ======================================================

    selected_evidence = generation.get(
        "selected_sentences",
        [],
    )

    # ======================================================
    # FINAL RESULT
    # ======================================================

    result = {
        "text": answer,

        "answer": answer,

        "rerank_score": safe_float(
            top_docs[0].get(
                "rerank_score",
                0.0,
            )
        ),

        "context_score": safe_float(
            top_docs[0].get(
                "context_score",
                0.0,
            )
        ),

        "pipeline_confidence": confidence,

        "retrieval_confidence": (
            retrieval_confidence
        ),

        "answer_confidence": (
            answer_confidence
        ),

        "answer_valid": answer_valid,

        "domain": top_docs[0].get(
            "domain",
            "general",
        ),

        "source": top_docs[0].get(
            "source",
            "unknown",
        ),

        "source_documents": (
            source_documents
        ),

        "evidence": evidence,

        "selected_evidence": (
            selected_evidence
        ),

        "candidate_sentences": (
            generation.get(
                "candidate_sentences",
                0,
            )
        ),

        "selected_sentences": (
            generation.get(
                "selected_count",
                len(
                    selected_evidence
                ),
            )
        ),

        "best_sentence_similarity": (
            generation.get(
                "best_similarity",
                0.0,
            )
        ),

        "best_evidence_score": (
            generation.get(
                "best_evidence_score",
                0.0,
            )
        ),

        "question_targets": (
            generation.get(
                "question_targets",
                [],
            )
        ),

        "embedding_reuse": (
            embedding_reuses
        ),

        "embedding_fallbacks": (
            embedding_fallbacks
        ),

        "fusion_sentence_count": (
            fusion_result.get(
                "sentence_count",
                0,
            )
        ),

        "fusion_graph": (
            fusion_result.get(
                "graph",
                [],
            )
        ),
    }

    # ======================================================
    # FINAL DEBUG
    # ======================================================

    print()
    print("=" * 70)
    print(
        "FINAL PIPELINE RESULT"
    )
    print("=" * 70)

    print()
    print(
        "Answer:",
        answer,
    )

    print()
    print(
        "Answer Words:",
        len(
            answer.split()
        ),
    )

    print(
        "Source Documents Used:",
        len(source_documents),
    )

    print(
        "Evidence Sentences:",
        len(selected_evidence),
    )

    print(
        "Pipeline Confidence:",
        confidence,
    )

    print(
        "Retrieval Confidence:",
        retrieval_confidence,
    )

    print(
        "Answer Confidence:",
        answer_confidence,
    )

    print(
        "Answer Valid:",
        answer_valid,
    )

    print()
    print(
        "Selected Evidence:"
    )

    for item in selected_evidence:

        print()
        print(
            f"[{item.get('rank')}] "
            f"similarity="
            f"{item.get('similarity', 0.0):.4f} "
            f"relevance="
            f"{item.get('relevance', 0.0):.4f} "
            f"targets="
            f"{item.get('targets', [])}"
        )

        print(
            item.get(
                "text",
                "",
            )
        )

    print()
    print(
        "[Optimization #17] "
        f"Embedding Reuse: "
        f"{embedding_reuses}"
    )

    print(
        "[Optimization #17] "
        f"Embedding Fallbacks: "
        f"{embedding_fallbacks}"
    )

    print("=" * 70)

    # The surrounding application historically expects a list.
    return [result]


# ==========================================================
# TEST / DIRECT EXECUTION
# ==========================================================

if __name__ == "__main__":

    result = run_pipeline(
        DEFAULT_QUERY
    )

    print()
    print(
        "======================================================================"
    )
    print(
        "RESULT"
    )
    print(
        "======================================================================"
    )

    for index, item in enumerate(
        result,
        start=1,
    ):

        print()
        print(
            f"--- Final Result {index} ---"
        )

        print(
            "Answer:",
            item.get(
                "text",
                "",
            )
        )

        print(
            "Pipeline Confidence:",
            item.get(
                "pipeline_confidence",
                0.0,
            )
        )

        print(
            "Retrieval Confidence:",
            item.get(
                "retrieval_confidence",
                0.0,
            )
        )

        print(
            "Answer Confidence:",
            item.get(
                "answer_confidence",
                0.0,
            )
        )

        print(
            "Answer Valid:",
            item.get(
                "answer_valid",
                False,
            )
        )

        print(
            "Source Documents:",
            item.get(
                "source_documents",
                [],
            )
        )

        print(
            "Candidate Sentences:",
            item.get(
                "candidate_sentences",
                0,
            )
        )

        print(
            "Selected Sentences:",
            item.get(
                "selected_sentences",
                0,
            )
        )

        print(
            "Question Targets:",
            item.get(
                "question_targets",
                [],
            )
        )

        print(
            "Selected Evidence:"
        )

        for evidence in item.get(
            "selected_evidence",
            [],
        ):

            print()

            print(
                f"[{evidence.get('rank')}] "
                f"similarity="
                f"{evidence.get('similarity', 0.0):.4f} "
                f"relevance="
                f"{evidence.get('relevance', 0.0):.4f}"
            )

            print(
                evidence.get(
                    "text",
                    "",
                )
            )

        print()

        print(
            "[Optimization #17] "
            "Embedding Reuse:",
            item.get(
                "embedding_reuse",
                0,
            )
        )

        print(
            "[Optimization #17] "
            "Embedding Fallbacks:",
            item.get(
                "embedding_fallbacks",
                0,
            )
        )