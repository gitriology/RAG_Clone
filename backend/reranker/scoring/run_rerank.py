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

CLI
---
python -m backend.reranker.scoring.run_rerank

python -m backend.reranker.scoring.run_rerank \
    --query "What is the Mars Orbiter Mission (MOM) and which organization developed it?"

Optional:
python -m backend.reranker.scoring.run_rerank \
    --query "What is the Mars Orbiter Mission (MOM) and which organization developed it?" \
    --top-k 3 \
    --max-sentences 3

Optimization #4
---------------
Reuse CrossEncoder scores produced by MS-ARC.

Optimization #17
----------------
Reuse sentence embeddings between Phase 10.1
and Phase 10.2.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional


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

import argparse
# ==========================================================
# CONFIGURATION
# ==========================================================

DEFAULT_TOP_K_DOCUMENTS = 3

DEFAULT_MAX_SENTENCES = 3

MIN_CONFIDENCE = 0.40

DEFAULT_QUERY = (
    "What is the Mars Orbiter Mission (MOM) "
    "and which organization developed it?"
)


# ==========================================================
# SAFE FLOAT
# ==========================================================

def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Safely convert a value to float.
    """

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


# ==========================================================
# ARGUMENT PARSER
# ==========================================================

def parse_args(
    argv: Optional[List[str]] = None,
) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Example
    -------
    python -m backend.reranker.scoring.run_rerank \
        --query "What is the Mars Orbiter Mission (MOM)?"
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run the production retrieval, reranking, "
            "evidence selection, fusion and validation pipeline."
        )
    )

    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help=(
            "Question to run through the RAG pipeline. "
            "If omitted, the default MOM question is used."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K_DOCUMENTS,
        help=(
            "Number of validated documents passed to "
            "the answer generation stage. "
            f"Default: {DEFAULT_TOP_K_DOCUMENTS}"
        ),
    )

    parser.add_argument(
        "--max-sentences",
        type=int,
        default=DEFAULT_MAX_SENTENCES,
        help=(
            "Maximum number of evidence sentences selected "
            "during question-aware generation. "
            f"Default: {DEFAULT_MAX_SENTENCES}"
        ),
    )

    return parser.parse_args(argv)


# ==========================================================
# QUERY NORMALIZATION
# ==========================================================

def normalize_query(
    query: Optional[str],
) -> str:
    """
    Normalize the incoming query.

    Empty CLI input falls back to DEFAULT_QUERY.
    """

    if query is None:
        return DEFAULT_QUERY

    normalized = query.strip()

    if not normalized:
        return DEFAULT_QUERY

    return normalized


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def run_pipeline(
    query: str,
    top_k_documents: int = DEFAULT_TOP_K_DOCUMENTS,
    max_sentences: int = DEFAULT_MAX_SENTENCES,
) -> List[Dict]:
    """
    Execute the complete production RAG pipeline.

    Parameters
    ----------
    query:
        User question.

    top_k_documents:
        Number of validated documents passed downstream.

    max_sentences:
        Maximum number of evidence sentences selected.
    """

    # ======================================================
    # QUERY
    # ======================================================

    query = normalize_query(query)

    if top_k_documents < 1:
        raise ValueError(
            "top_k_documents must be >= 1."
        )

    if max_sentences < 1:
        raise ValueError(
            "max_sentences must be >= 1."
        )

    print()
    print("=" * 70)
    print("RERANKING / ANSWER PIPELINE")
    print("=" * 70)

    print()
    print(
        "Query:",
        query,
    )

    print(
        "Top-K Documents:",
        top_k_documents,
    )

    print(
        "Max Evidence Sentences:",
        max_sentences,
    )

    # ======================================================
    # STEP 1
    # RETRIEVAL + MS-ARC
    # ======================================================

    retrieval_result = retrieve(
        query
    )

    if not isinstance(
        retrieval_result,
        dict,
    ):
        raise RuntimeError(
            "Retrieval pipeline returned "
            "an invalid result."
        )

    retrieved_docs = retrieval_result.get(
        "documents",
        [],
    )

    if not isinstance(
        retrieved_docs,
        list,
    ):
        retrieved_docs = []

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
    # NO DOCUMENTS
    # ======================================================

    if not retrieved_docs:

        print()
        print(
            "[Pipeline] No documents retrieved."
        )

        return []

    # ======================================================
    # OPTIMIZATION #4
    # VERIFY MS-ARC SCORES
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

    if not isinstance(
        ranked_docs,
        list,
    ):
        ranked_docs = []

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

    if not isinstance(
        validated_docs,
        list,
    ):
        validated_docs = []

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
        k=top_k_documents,
    )

    if not top_docs:

        print()
        print(
            "No usable documents."
        )

        return []

    print(
        "Top Docs:",
        len(top_docs),
    )

    # ======================================================
    # PRINT TOP DOCUMENT DEBUG
    # ======================================================

    print()
    print(
        "[Pipeline] Final Context Documents:"
    )

    for rank, doc in enumerate(
        top_docs,
        start=1,
    ):

        print()

        print(
            f"[{rank}] "
            f"doc_id={doc.get('doc_id')} "
            f"source={doc.get('source')} "
            f"domain={doc.get('domain')}"
        )

        print(
            f"    rerank_score="
            f"{safe_float(doc.get('rerank_score')):.4f}"
        )

        print(
            f"    context_score="
            f"{safe_float(doc.get('context_score')):.4f}"
        )

    # ======================================================
    # STEP 5
    # QUESTION-AWARE GENERATION
    # ======================================================

    generation = generate_answer(
        query=query,
        documents=top_docs,
        max_sentences=max_sentences,
    )

    if not isinstance(
        generation,
        dict,
    ):
        generation = {}

    # ======================================================
    # OPTIMIZATION #17
    # REUSABLE SENTENCE EMBEDDINGS
    # ======================================================

    sentence_embeddings = generation.get(
        "sentence_embeddings",
        {},
    )

    if not isinstance(
        sentence_embeddings,
        dict,
    ):
        sentence_embeddings = {}

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

    if not isinstance(
        fusion_result,
        dict,
    ):
        fusion_result = {}

    answer = fusion_result.get(
        "paragraph",
        "",
    )

    if not isinstance(
        answer,
        str,
    ):
        answer = str(answer)

    answer = answer.strip()

    # ======================================================
    # FALLBACK TO PHASE 10.1
    # ======================================================

    if not answer:

        answer = generation.get(
            "answer",
            "",
        )

        if not isinstance(
            answer,
            str,
        ):
            answer = str(answer)

        answer = answer.strip()

    # ======================================================
    # OPTIMIZATION #17 DEBUG
    # ======================================================

    embedding_reuses = int(
        safe_float(
            fusion_result.get(
                "embedding_reuses",
                0,
            )
        )
    )

    embedding_fallbacks = int(
        safe_float(
            fusion_result.get(
                "embedding_fallbacks",
                0,
            )
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

    if not isinstance(
        validation,
        dict,
    ):
        validation = {}

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

        print()
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

    if not isinstance(
        selected_evidence,
        list,
    ):
        selected_evidence = []

    # ======================================================
    # FINAL RESULT
    # ======================================================

    result = {
        "text": answer,

        "answer": answer,

        "query": query,

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
        "Query:",
        query,
    )

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
            f"{safe_float(item.get('similarity')):.4f} "
            f"relevance="
            f"{safe_float(item.get('relevance')):.4f} "
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

    parser = argparse.ArgumentParser(
        description=(
            "Run the production RAG retrieval, "
            "reranking, evidence selection, "
            "and answer validation pipeline."
        )
    )

    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help=(
            "Question to send through the RAG pipeline."
        ),
    )

    args = parser.parse_args()

    result = run_pipeline(
        args.query
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

        print()
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