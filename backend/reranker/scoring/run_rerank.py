from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional


from backend.retrieval.run_retrieval import (
    retrieve,
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

from backend.evidence_graph.graph_builder import (
    build_graph,
)

from backend.evidence_state.build_evidence_state import (
    build_evidence_state,
)


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

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


# ==========================================================
# SAFE INT
# ==========================================================

def safe_int(
    value: Any,
    default: int = 0,
) -> int:

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


# ==========================================================
# SAFE LIST
# ==========================================================

def safe_list(
    value: Any,
) -> List:

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, set):
        return list(value)

    # NumPy / Torch / tensor-like values.
    try:

        if hasattr(value, "detach"):

            value = value.detach()

        if hasattr(value, "cpu"):

            value = value.cpu()

        if hasattr(value, "tolist"):

            converted = value.tolist()

            if isinstance(converted, list):
                return converted

            return [converted]

    except Exception:
        pass

    try:
        return list(value)

    except TypeError:
        return []


# ==========================================================
# SAFE DICT
# ==========================================================

def safe_dict(
    value: Any,
) -> Dict:

    if isinstance(value, dict):
        return value

    return {}


# ==========================================================
# GENERIC GETTER
# ==========================================================

def get_value(
    obj: Any,
    key: str,
    default: Any = None,
) -> Any:

    if obj is None:
        return default

    if isinstance(obj, dict):

        return obj.get(
            key,
            default,
        )

    return getattr(
        obj,
        key,
        default,
    )


# ==========================================================
# DOCUMENT GETTER
# ==========================================================

def get_document_value(
    document: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    Read a document value from either:

        dict["field"]

    or:

        object.field
    """

    return get_value(
        document,
        key,
        default,
    )


# ==========================================================
# DOCUMENT METADATA
# ==========================================================

def get_document_metadata(
    document: Any,
) -> Dict[str, Any]:

    metadata = get_document_value(
        document,
        "metadata",
        {},
    )

    if isinstance(
        metadata,
        dict,
    ):
        return metadata

    return {}


# ==========================================================
# VECTOR CONVERSION
# ==========================================================

def normalize_vector(
    value: Any,
) -> List[Any]:
    """
    Convert NumPy / Torch / tuple-like vectors into
    a normal Python list.

    This is important because the production integration
    test needs the actual evidence vector to be available
    in the pipeline result.
    """

    if value is None:
        return []

    # Torch tensor.
    try:

        if hasattr(value, "detach"):

            value = value.detach()

        if hasattr(value, "cpu"):

            value = value.cpu()

        if hasattr(value, "tolist"):

            converted = value.tolist()

            if isinstance(
                converted,
                list,
            ):
                return converted

            return [converted]

    except Exception:
        pass

    # NumPy array.
    try:

        if hasattr(value, "tolist"):

            converted = value.tolist()

            if isinstance(
                converted,
                list,
            ):
                return converted

            return [converted]

    except Exception:
        pass

    # Normal Python containers.
    if isinstance(
        value,
        list,
    ):
        return value

    if isinstance(
        value,
        tuple,
    ):
        return list(value)

    try:
        return list(value)

    except TypeError:
        return []


# ==========================================================
# VECTOR DIMENSIONS
# ==========================================================

def vector_length(
    value: Any,
) -> int:

    vector = normalize_vector(
        value
    )

    return len(vector)


# ==========================================================
# ARGUMENT PARSER
# ==========================================================

def parse_args(
    argv: Optional[List[str]] = None,
) -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run the production retrieval, MS-ARC reranking, "
            "Evidence Graph, Evidence State, evidence selection, "
            "fusion and validation pipeline."
        )
    )

    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help=(
            "Question to run through the RAG pipeline."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K_DOCUMENTS,
        help=(
            "Number of validated documents passed to "
            "answer generation."
        ),
    )

    parser.add_argument(
        "--max-sentences",
        type=int,
        default=DEFAULT_MAX_SENTENCES,
        help=(
            "Maximum number of evidence sentences."
        ),
    )

    return parser.parse_args(argv)


# ==========================================================
# QUERY NORMALIZATION
# ==========================================================

def normalize_query(
    query: Optional[str],
) -> str:

    if query is None:
        return DEFAULT_QUERY

    normalized = query.strip()

    if not normalized:
        return DEFAULT_QUERY

    return normalized


# ==========================================================
# EVIDENCE GRAPH RESULT
# ==========================================================

def build_evidence_graph_result(
    graph_state: Any,
) -> Dict[str, Any]:

    nodes = safe_list(
        get_value(
            graph_state,
            "nodes",
            [],
        )
    )

    edges = safe_list(
        get_value(
            graph_state,
            "edges",
            [],
        )
    )

    graph = get_value(
        graph_state,
        "graph",
        None,
    )

    if graph is not None:

        try:
            node_count = graph.number_of_nodes()

        except AttributeError:
            node_count = len(nodes)

        try:
            edge_count = graph.number_of_edges()

        except AttributeError:
            edge_count = len(edges)

    else:

        node_count = len(nodes)
        edge_count = len(edges)

    graph_score = safe_float(
        get_value(
            graph_state,
            "graph_score",
            0.0,
        )
    )

    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "graph_score": graph_score,

        # Backwards compatibility.
        "nodes": node_count,
        "edges": edge_count,
        "score": graph_score,
    }


# ==========================================================
# EVIDENCE STATE RESULT
# ==========================================================

def build_evidence_state_result(
    evidence_state: Any,
    fallback_evidence: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Convert the EvidenceState object into the public
    pipeline result.

    IMPORTANT:

    The actual evidence vector is preserved.

    We expose BOTH:

        evidence_vector
        evidence_weighted_vector

    as well as their dimensions.

    This prevents the production integration test from
    receiving:

        evidence_vector = None

    while only receiving:

        vector_dimensions = 25
    """

    fallback_evidence = safe_dict(
        fallback_evidence
    )

    # ======================================================
    # FEATURES
    # ======================================================

    features = get_value(
        evidence_state,
        "features",
        None,
    )

    if features is None:

        features = fallback_evidence.get(
            "features",
            [],
        )

    if isinstance(
        features,
        dict,
    ):

        feature_count = len(
            features
        )

    elif isinstance(
        features,
        (list, tuple, set),
    ):

        feature_count = len(
            features
        )

    else:

        feature_count = safe_int(
            get_value(
                evidence_state,
                "feature_count",
                fallback_evidence.get(
                    "feature_count",
                    0,
                ),
            )
        )

    # ======================================================
    # GROUPS
    # ======================================================

    groups = get_value(
        evidence_state,
        "groups",
        None,
    )

    if groups is None:

        groups = fallback_evidence.get(
            "groups",
            [],
        )

    if isinstance(
        groups,
        dict,
    ):

        group_count = len(
            groups
        )

    elif isinstance(
        groups,
        (list, tuple, set),
    ):

        group_count = len(
            groups
        )

    else:

        group_count = safe_int(
            get_value(
                evidence_state,
                "group_count",
                fallback_evidence.get(
                    "group_count",
                    0,
                ),
            )
        )

    # ======================================================
    # ACTUAL EVIDENCE VECTOR
    # ======================================================

    evidence_vector_raw = get_value(
        evidence_state,
        "vector",
        None,
    )

    if evidence_vector_raw is None:

        evidence_vector_raw = get_value(
            evidence_state,
            "evidence_vector",
            None,
        )

    if evidence_vector_raw is None:

        evidence_vector_raw = fallback_evidence.get(
            "evidence_vector",
            fallback_evidence.get(
                "vector",
                [],
            ),
        )

    evidence_vector = normalize_vector(
        evidence_vector_raw
    )

    # ======================================================
    # ACTUAL WEIGHTED VECTOR
    # ======================================================

    weighted_vector_raw = get_value(
        evidence_state,
        "weighted_vector",
        None,
    )

    if weighted_vector_raw is None:

        weighted_vector_raw = get_value(
            evidence_state,
            "evidence_weighted_vector",
            None,
        )

    if weighted_vector_raw is None:

        weighted_vector_raw = fallback_evidence.get(
            "evidence_weighted_vector",
            fallback_evidence.get(
                "weighted_vector",
                [],
            ),
        )

    weighted_vector = normalize_vector(
        weighted_vector_raw
    )

    # ======================================================
    # VECTOR DIMENSIONS
    # ======================================================

    vector_dimensions = vector_length(
        evidence_vector
    )

    if vector_dimensions == 0:

        vector_dimensions = safe_int(
            get_value(
                evidence_state,
                "vector_dimensions",
                fallback_evidence.get(
                    "vector_dimensions",
                    0,
                ),
            )
        )

    weighted_vector_dimensions = vector_length(
        weighted_vector
    )

    if weighted_vector_dimensions == 0:

        weighted_vector_dimensions = safe_int(
            get_value(
                evidence_state,
                "weighted_vector_dimensions",
                fallback_evidence.get(
                    "weighted_vector_dimensions",
                    0,
                ),
            )
        )

    # ======================================================
    # EVIDENCE SCORE
    # ======================================================

    evidence_score = safe_float(
        get_value(
            evidence_state,
            "evidence_score",
            fallback_evidence.get(
                "evidence_score",
                0.0,
            ),
        )
    )

    # ======================================================
    # PROFILE SCORE
    # ======================================================

    profile_score = safe_float(
        get_value(
            evidence_state,
            "profile_score",
            fallback_evidence.get(
                "profile_score",
                0.0,
            ),
        )
    )

    # ======================================================
    # HEALTH SCORE
    # ======================================================

    health_score = safe_float(
        get_value(
            evidence_state,
            "health_score",
            fallback_evidence.get(
                "health_score",
                0.0,
            ),
        )
    )

    # ======================================================
    # UNCERTAINTY SCORE
    # ======================================================

    uncertainty_score = safe_float(
        get_value(
            evidence_state,
            "uncertainty_score",
            fallback_evidence.get(
                "uncertainty_score",
                0.0,
            ),
        )
    )

    # ======================================================
    # RESULT
    # ======================================================

    return {
        # Counts
        "feature_count": feature_count,
        "group_count": group_count,

        # ACTUAL vectors
        "evidence_vector": evidence_vector,
        "vector": evidence_vector,

        "evidence_weighted_vector": weighted_vector,
        "weighted_vector": weighted_vector,

        # Dimensions
        "vector_dimensions": vector_dimensions,
        "weighted_vector_dimensions": (
            weighted_vector_dimensions
        ),

        # Scores
        "evidence_score": evidence_score,
        "profile_score": profile_score,
        "health_score": health_score,
        "uncertainty_score": uncertainty_score,
    }


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def run_pipeline(
    query: str,
    top_k_documents: int = DEFAULT_TOP_K_DOCUMENTS,
    max_sentences: int = DEFAULT_MAX_SENTENCES,
) -> List[Dict]:

    query = normalize_query(
        query
    )

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
    print("Query:", query)

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
            "Retrieval pipeline returned an invalid result."
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

    retrieval_state = retrieval_result.get(
        "retrieval_state"
    )

    if retrieval_state is None:

        raise RuntimeError(
            "Optimization #21 failed: "
            "MS-ARC RetrievalState was not "
            "preserved by retrieval."
        )

    if not hasattr(
        retrieval_state,
        "reranked_results",
    ):

        raise RuntimeError(
            "Optimization #21 failed: "
            "RetrievalState does not expose "
            "'reranked_results'."
        )

    reranked_results = (
        retrieval_state.reranked_results
    )

    if reranked_results is None:

        raise RuntimeError(
            "Optimization #21 failed: "
            "RetrievalState.reranked_results is None."
        )

    if not isinstance(
        reranked_results,
        (list, tuple),
    ):

        raise RuntimeError(
            "Optimization #21 failed: "
            "RetrievalState.reranked_results "
            "must be a list or tuple."
        )

    if not reranked_results:

        raise RuntimeError(
            "Optimization #21 failed: "
            "RetrievalState.reranked_results is empty."
        )

    print()
    print(
        "[Optimization #21] "
        "MS-ARC RetrievalState available."
    )

    print(
        "[Optimization #21] "
        f"Reranked results preserved: "
        f"{len(reranked_results)}"
    )

    retrieval_confidence = safe_float(
        retrieval_result.get(
            "retrieval_confidence",
            0.0,
        )
    )

    evidence = safe_dict(
        retrieval_result.get(
            "evidence",
            {},
        )
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
        if get_document_value(
            doc,
            "rerank_score",
            None,
        ) is None
    ]

    if missing_scores:

        raise RuntimeError(
            "Optimization #4 failed: "
            "MS-ARC did not propagate "
            "rerank_score to downstream documents."
        )

    print()
    print(
        "[Optimization #4] "
        "MS-ARC rerank scores available."
    )

    # ======================================================
    # STEP 2
    # EVIDENCE GRAPH
    # ======================================================

    print()
    print("=" * 70)
    print(
        "OPTIMIZATION #20 / #21 : EVIDENCE GRAPH"
    )
    print("=" * 70)

    # IMPORTANT:
    #
    # Do NOT rerank here.
    #
    # Existing MS-ARC RetrievalState is reused.

    graph_state = build_graph(
        retrieval_state
    )

    if graph_state is None:

        raise RuntimeError(
            "Evidence Graph builder returned None."
        )

    graph_result = build_evidence_graph_result(
        graph_state
    )

    print(
        "[Optimization #21] "
        f"Graph nodes: "
        f"{graph_result['node_count']}"
    )

    print(
        "[Optimization #21] "
        f"Graph edges: "
        f"{graph_result['edge_count']}"
    )

    print(
        "[Optimization #20] "
        "Optimized semantic edge construction: ACTIVE"
    )

    print(
        "[Optimization #21] "
        "Evidence Graph integration: PASS"
    )

    print("=" * 70)

    # ======================================================
    # STEP 3
    # EVIDENCE STATE
    # ======================================================

    print()
    print("=" * 70)
    print(
        "EVIDENCE STATE"
    )
    print("=" * 70)

    if retrieval_state is None:

        raise RuntimeError(
            "Evidence State construction failed: "
            "retrieval_state is None."
        )

    if graph_state is None:

        raise RuntimeError(
            "Evidence State construction failed: "
            "graph_state is None."
        )

    # ------------------------------------------------------
    # IMPORTANT:
    #
    # Build EvidenceState EXACTLY ONCE.
    #
    # The current production builder requires:
    #
    #     build_evidence_state(
    #         retrieval_state,
    #         graph_state,
    #     )
    #
    # Do NOT use a try/except TypeError fallback here.
    #
    # A TypeError raised INSIDE the builder must not cause
    # the builder to execute a second time.
    # ------------------------------------------------------

    evidence_state = build_evidence_state(
        retrieval_state,
        graph_state,
    )

    if evidence_state is None:

        raise RuntimeError(
            "Evidence State builder returned None."
        )

    evidence_state_result = (
        build_evidence_state_result(
            evidence_state,
            fallback_evidence=evidence,
        )
    )

    # ======================================================
    # VALIDATE ACTUAL VECTOR
    # ======================================================

    evidence_vector = evidence_state_result.get(
        "evidence_vector",
        [],
    )

    weighted_evidence_vector = (
        evidence_state_result.get(
            "evidence_weighted_vector",
            [],
        )
    )

    evidence_vector_dimensions = (
        vector_length(
            evidence_vector
        )
    )

    weighted_vector_dimensions = (
        vector_length(
            weighted_evidence_vector
        )
    )

    # ------------------------------------------------------
    # If the builder reports dimensions but the actual
    # vector is absent, do not silently fabricate values.
    #
    # This catches an implementation mismatch immediately.
    # ------------------------------------------------------

    reported_vector_dimensions = safe_int(
        evidence_state_result.get(
            "vector_dimensions",
            0,
        )
    )

    if (
        evidence_vector_dimensions == 0
        and reported_vector_dimensions > 0
    ):

        # Preserve reported dimension for diagnostics,
        # but explicitly expose that the actual vector
        # was unavailable.
        print(
            "[Evidence State] "
            "WARNING: reported vector dimensions =",
            reported_vector_dimensions,
            "but actual evidence vector is empty."
        )

    else:

        evidence_state_result[
            "vector_dimensions"
        ] = evidence_vector_dimensions

    if weighted_vector_dimensions > 0:

        evidence_state_result[
            "weighted_vector_dimensions"
        ] = weighted_vector_dimensions

    print(
        "[Evidence State] "
        "EvidenceState construction: PASS"
    )

    print(
        "[Evidence State] Features:",
        evidence_state_result[
            "feature_count"
        ],
    )

    print(
        "[Evidence State] Groups:",
        evidence_state_result[
            "group_count"
        ],
    )

    print(
        "[Evidence State] Vector dimensions:",
        evidence_state_result[
            "vector_dimensions"
        ],
    )

    print(
        "[Evidence State] Weighted vector dimensions:",
        evidence_state_result[
            "weighted_vector_dimensions"
        ],
    )

    print(
        "[Evidence State] Evidence vector length:",
        len(
            evidence_state_result[
                "evidence_vector"
            ]
        ),
    )

    print(
        "[Evidence State] Weighted evidence vector length:",
        len(
            evidence_state_result[
                "evidence_weighted_vector"
            ]
        ),
    )

    print(
        "[Evidence State] Evidence score:",
        f"{evidence_state_result['evidence_score']:.4f}",
    )

    print(
        "[Evidence State] Profile score:",
        f"{evidence_state_result['profile_score']:.4f}",
    )

    print(
        "[Evidence State] Health score:",
        f"{evidence_state_result['health_score']:.4f}",
    )

    print(
        "[Evidence State] Uncertainty score:",
        f"{evidence_state_result['uncertainty_score']:.4f}",
    )

    print("=" * 70)

    # ======================================================
    # STEP 4
    # USE EXISTING MS-ARC RESULTS
    # ======================================================

    ranked_docs = retrieved_docs

    print()
    print(
        "[MS-ARC] "
        "Reusing existing reranked documents."
    )

    print(
        "[MS-ARC] Ranked Docs:",
        len(ranked_docs),
    )

    # ======================================================
    # STEP 5
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
    # STEP 6
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
            f"doc_id="
            f"{get_document_value(doc, 'doc_id')} "
            f"source="
            f"{get_document_value(doc, 'source')} "
            f"domain="
            f"{get_document_value(doc, 'domain')}"
        )

        print(
            "    rerank_score="
            f"{safe_float(get_document_value(doc, 'rerank_score')):.4f}"
        )

        print(
            "    context_score="
            f"{safe_float(get_document_value(doc, 'context_score')):.4f}"
        )

    # ======================================================
    # STEP 7
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
    # STEP 7.1
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

    embedding_reuses = safe_int(
        fusion_result.get(
            "embedding_reuses",
            0,
        )
    )

    embedding_fallbacks = safe_int(
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
    # STEP 8
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
    # STEP 9
    # PIPELINE CONFIDENCE
    # ======================================================

    confidence = safe_float(
        compute_confidence(
            documents=top_docs,
            validation=validation,
            retrieval_confidence=retrieval_confidence,
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

        source = get_document_value(
            doc,
            "source",
            "unknown",
        )

        doc_id = get_document_value(
            doc,
            "doc_id",
        )

        domain = get_document_value(
            doc,
            "domain",
            None,
        )

        if domain is None:

            metadata = get_document_metadata(
                doc
            )

            domain = metadata.get(
                "domain",
                "general",
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
                "domain": domain,
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
    # FINAL DOCUMENT METADATA
    # ======================================================

    top_document = top_docs[0]

    top_domain = get_document_value(
        top_document,
        "domain",
        None,
    )

    if top_domain is None:

        top_domain = get_document_metadata(
            top_document
        ).get(
            "domain",
            "general",
        )

    top_source = get_document_value(
        top_document,
        "source",
        None,
    )

    if top_source is None:

        top_source = get_document_metadata(
            top_document
        ).get(
            "source",
            "unknown",
        )

    # ======================================================
    # ACTUAL EVIDENCE VECTORS
    # ======================================================

    final_evidence_vector = normalize_vector(
        evidence_state_result.get(
            "evidence_vector",
            [],
        )
    )

    final_weighted_evidence_vector = (
        normalize_vector(
            evidence_state_result.get(
                "evidence_weighted_vector",
                [],
            )
        )
    )

    final_vector_dimensions = len(
        final_evidence_vector
    )

    final_weighted_vector_dimensions = len(
        final_weighted_evidence_vector
    )

    # ======================================================
    # FINAL RESULT
    # ======================================================

    result = {

        "text": answer,

        "answer": answer,

        "query": query,

        # ==================================================
        # Ranking
        # ==================================================

        "rerank_score": safe_float(
            get_document_value(
                top_document,
                "rerank_score",
                0.0,
            )
        ),

        "context_score": safe_float(
            get_document_value(
                top_document,
                "context_score",
                0.0,
            )
        ),

        # ==================================================
        # Confidence
        # ==================================================

        "pipeline_confidence": confidence,

        "retrieval_confidence": (
            retrieval_confidence
        ),

        "answer_confidence": (
            answer_confidence
        ),

        "answer_valid": answer_valid,

        # ==================================================
        # Source
        # ==================================================

        "domain": top_domain,

        "source": top_source,

        "source_documents": source_documents,

        # ==================================================
        # Retrieval evidence
        # ==================================================

        "evidence": evidence,

        # ==================================================
        # Selected evidence
        # ==================================================

        "selected_evidence": selected_evidence,

        "candidate_sentences": generation.get(
            "candidate_sentences",
            0,
        ),

        "selected_sentences": generation.get(
            "selected_count",
            len(selected_evidence),
        ),

        "best_sentence_similarity": generation.get(
            "best_similarity",
            0.0,
        ),

        "best_evidence_score": generation.get(
            "best_evidence_score",
            0.0,
        ),

        "question_targets": generation.get(
            "question_targets",
            [],
        ),

        # ==================================================
        # Optimization #17
        # ==================================================

        "embedding_reuse": embedding_reuses,

        "embedding_fallbacks": embedding_fallbacks,

        "fusion_sentence_count": fusion_result.get(
            "sentence_count",
            0,
        ),

        "fusion_graph": fusion_result.get(
            "graph",
            [],
        ),

        # ==================================================
        # Optimization #20 / #21
        # ==================================================

        "evidence_graph": graph_result,

        "graph_nodes": graph_result[
            "node_count"
        ],

        "graph_edges": graph_result[
            "edge_count"
        ],

        # ==================================================
        # Evidence State
        # ==================================================

        "evidence_state": evidence_state_result,

        # Counts
        "evidence_features": (
            evidence_state_result[
                "feature_count"
            ]
        ),

        "evidence_groups": (
            evidence_state_result[
                "group_count"
            ]
        ),

        # ==================================================
        # ACTUAL EVIDENCE VECTOR
        #
        # These are the fields the production integration
        # test can consume directly.
        # ==================================================

        "evidence_vector": final_evidence_vector,

        "evidence_weighted_vector": (
            final_weighted_evidence_vector
        ),

        # Backwards-compatible aliases.

        "vector": final_evidence_vector,

        "weighted_vector": (
            final_weighted_evidence_vector
        ),

        # ==================================================
        # VECTOR DIMENSIONS
        # ==================================================

        "evidence_vector_dimensions": (
            final_vector_dimensions
        ),

        "evidence_weighted_vector_dimensions": (
            final_weighted_vector_dimensions
        ),

        "vector_dimensions": (
            final_vector_dimensions
        ),

        "weighted_vector_dimensions": (
            final_weighted_vector_dimensions
        ),

        # ==================================================
        # Evidence Scores
        # ==================================================

        "evidence_score": (
            evidence_state_result[
                "evidence_score"
            ]
        ),

        "evidence_profile_score": (
            evidence_state_result[
                "profile_score"
            ]
        ),

        "evidence_health_score": (
            evidence_state_result[
                "health_score"
            ]
        ),

        "evidence_uncertainty_score": (
            evidence_state_result[
                "uncertainty_score"
            ]
        ),
    }

    # ======================================================
    # FINAL DEBUG
    # ======================================================

    print()
    print("=" * 70)
    print("FINAL PIPELINE RESULT")
    print("=" * 70)

    print()
    print("Query:", query)

    print()
    print("Answer:", answer)

    print(
        "Answer Words:",
        len(answer.split()),
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

    # ======================================================
    # EVIDENCE STATE DEBUG
    # ======================================================

    print()
    print("Evidence State:")

    print(
        "Evidence Features:",
        evidence_state_result[
            "feature_count"
        ],
    )

    print(
        "Evidence Groups:",
        evidence_state_result[
            "group_count"
        ],
    )

    print(
        "Evidence Vector Dimensions:",
        final_vector_dimensions,
    )

    print(
        "Weighted Vector Dimensions:",
        final_weighted_vector_dimensions,
    )

    print(
        "Evidence Vector:",
        final_evidence_vector,
    )

    print(
        "Weighted Evidence Vector:",
        final_weighted_evidence_vector,
    )

    print(
        "Evidence Score:",
        evidence_state_result[
            "evidence_score"
        ],
    )

    print(
        "Evidence Profile Score:",
        evidence_state_result[
            "profile_score"
        ],
    )

    print(
        "Evidence Health Score:",
        evidence_state_result[
            "health_score"
        ],
    )

    print(
        "Evidence Uncertainty Score:",
        evidence_state_result[
            "uncertainty_score"
        ],
    )

    # ======================================================
    # EVIDENCE GRAPH DEBUG
    # ======================================================

    print()
    print("Evidence Graph:")

    print(
        "Graph Nodes:",
        graph_result[
            "node_count"
        ],
    )

    print(
        "Graph Edges:",
        graph_result[
            "edge_count"
        ],
    )

    print(
        "Graph Score:",
        graph_result[
            "graph_score"
        ],
    )

    # ======================================================
    # SELECTED EVIDENCE
    # ======================================================

    print()
    print("Selected Evidence:")

    for item in selected_evidence:

        if not isinstance(
            item,
            dict,
        ):

            print(item)
            continue

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

    # ======================================================
    # OPTIMIZATION DEBUG
    # ======================================================

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

    print()
    print(
        "[Optimization #20] "
        "Evidence Graph semantic batching: ACTIVE"
    )

    print(
        "[Optimization #21] "
        "RetrievalState reuse: ACTIVE"
    )

    print(
        "[Evidence State] "
        "EvidenceState construction: ACTIVE"
    )

    print(
        "[Evidence State] "
        f"Actual vector length: "
        f"{len(final_evidence_vector)}"
    )

    print(
        "[Evidence State] "
        f"Actual weighted vector length: "
        f"{len(final_weighted_evidence_vector)}"
    )

    print("=" * 70)

    return [result]


# ==========================================================
# DIRECT EXECUTION
# ==========================================================

if __name__ == "__main__":

    args = parse_args()

    result = run_pipeline(
        query=args.query,
        top_k_documents=args.top_k,
        max_sentences=args.max_sentences,
    )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

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

        # ==================================================
        # EVIDENCE GRAPH
        # ==================================================

        graph = item.get(
            "evidence_graph",
            {},
        )

        print()
        print(
            "Evidence Graph:",
            graph,
        )

        print(
            "Graph Nodes:",
            graph.get(
                "node_count",
                0,
            ),
        )

        print(
            "Graph Edges:",
            graph.get(
                "edge_count",
                0,
            ),
        )

        print(
            "Graph Score:",
            graph.get(
                "graph_score",
                0.0,
            ),
        )

        # ==================================================
        # EVIDENCE STATE
        # ==================================================

        evidence_state = item.get(
            "evidence_state",
            {},
        )

        print()
        print(
            "Evidence State:",
            evidence_state,
        )

        print(
            "Evidence Features:",
            evidence_state.get(
                "feature_count",
                0,
            ),
        )

        print(
            "Evidence Groups:",
            evidence_state.get(
                "group_count",
                0,
            ),
        )

        # ==================================================
        # ACTUAL VECTOR
        # ==================================================

        evidence_vector = item.get(
            "evidence_vector",
            [],
        )

        weighted_vector = item.get(
            "evidence_weighted_vector",
            [],
        )

        print(
            "Evidence Vector:",
            evidence_vector,
        )

        print(
            "Evidence Vector Length:",
            len(evidence_vector),
        )

        print(
            "Evidence Vector Dimensions:",
            item.get(
                "evidence_vector_dimensions",
                len(evidence_vector),
            ),
        )

        print(
            "Weighted Evidence Vector:",
            weighted_vector,
        )

        print(
            "Weighted Evidence Vector Length:",
            len(weighted_vector),
        )

        print(
            "Weighted Vector Dimensions:",
            item.get(
                "evidence_weighted_vector_dimensions",
                len(weighted_vector),
            ),
        )

        print(
            "Evidence Score:",
            evidence_state.get(
                "evidence_score",
                0.0,
            ),
        )

        print()
        print(
            "Selected Evidence:"
        )

        for evidence_item in item.get(
            "selected_evidence",
            [],
        ):

            if not isinstance(
                evidence_item,
                dict,
            ):

                print(evidence_item)
                continue

            print()

            print(
                f"[{evidence_item.get('rank')}] "
                f"similarity="
                f"{safe_float(evidence_item.get('similarity')):.4f} "
                f"relevance="
                f"{safe_float(evidence_item.get('relevance')):.4f}"
            )

            print(
                evidence_item.get(
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

        print()

        print(
            "[Optimization #20] "
            "Evidence Graph:",
            item.get(
                "evidence_graph",
                {},
            )
        )

        print(
            "[Optimization #21] "
            "RetrievalState reuse: ACTIVE"
        )

        print(
            "[Evidence State] "
            "EvidenceState construction: ACTIVE"
        )

        print(
            "[Evidence State] "
            "Evidence vector length:",
            len(
                item.get(
                    "evidence_vector",
                    [],
                )
            ),
        )

        print(
            "[Evidence State] "
            "Weighted vector length:",
            len(
                item.get(
                    "evidence_weighted_vector",
                    [],
                )
            ),
        )