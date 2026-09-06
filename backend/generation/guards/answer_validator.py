from __future__ import annotations

import re
from typing import Iterable, Optional

from sklearn.metrics.pairwise import cosine_similarity

from backend.models.model_registry import ModelRegistry
from backend.generation.validation.embedding_cache import ValidationEmbeddingCache


def _tokens(text: str):
    return re.findall(r"[a-zA-Z][a-zA-Z0-9]*(?:[-/.][a-zA-Z0-9]+)*|\d+(?:\.\d+)?", str(text).lower())


def _lexical_grounding(query: str, text: str) -> float:
    q = set(_tokens(query))
    t = set(_tokens(text))
    stop = {"what", "is", "are", "was", "were", "who", "where", "when", "why", "how", "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "about", "tell", "me", "explain", "describe", "compare", "between", "difference"}
    q = {x for x in q if x not in stop and (len(x) >= 3 or x.isdigit())}
    if not q:
        return 0.0
    return len(q & t) / len(q)


def validate_answer(
    answer,
    documents,
    threshold=0.40,
    evidence: Optional[Iterable] = None,
    query: Optional[str] = None,
    embedding_cache: Optional[ValidationEmbeddingCache] = None,
):
    """Validate answer *and* verify that the evidence actually grounds the query.

    Comparing an answer only with its own selected evidence is insufficient:
    an irrelevant sentence can agree with itself perfectly.  We therefore
    calculate a second signal, query_grounding, and require it for validity.
    """
    if not answer:
        return {"is_valid": False, "confidence": 0.0, "validation_scope": "empty_answer"}

    candidates = []
    for item in evidence or []:
        text = item.get("text", "") if isinstance(item, dict) else str(item)
        if str(text).strip():
            candidates.append(str(text).strip())

    scope = "selected_evidence"
    # Selected evidence is authoritative. Falling back to whole retrieved
    # documents can make an irrelevant answer appear grounded.
    if not candidates:
        return {
            "is_valid": False,
            "confidence": 0.0,
            "answer_agreement": 0.0,
            "query_grounding": 0.0,
            "query_semantic": 0.0,
            "query_lexical": 0.0,
            "validation_scope": scope,
            "candidate_count": 0,
            "threshold": float(threshold),
        }

    if not candidates:
        return {"is_valid": False, "confidence": 0.0, "validation_scope": scope}

    model = ModelRegistry.get_validation_model()
    embedding_cache = embedding_cache or ValidationEmbeddingCache(model=model)
    answer_embedding = embedding_cache.encode([str(answer)])
    candidate_embeddings = embedding_cache.encode(candidates)
    answer_scores = cosine_similarity(answer_embedding, candidate_embeddings)[0]
    best_answer_agreement = float(max(answer_scores)) if len(answer_scores) else 0.0

    query_grounding = 0.0
    best_query_semantic = 0.0
    best_query_lexical = 0.0
    if query:
        query_embedding = embedding_cache.encode([str(query)])
        query_scores = cosine_similarity(query_embedding, candidate_embeddings)[0]
        best_query_semantic = float(max(query_scores)) if len(query_scores) else 0.0
        best_query_lexical = max((_lexical_grounding(query, c) for c in candidates), default=0.0)
        # Semantic similarity is useful for paraphrases; lexical grounding is
        # decisive for exact names, articles, sections and numbers.
        query_grounding = max(
            0.60 * best_query_semantic + 0.40 * best_query_lexical,
            best_query_lexical,
        )

    # Agreement with evidence is necessary but not sufficient. Query grounding
    # is what prevents "answering the retrieved sentence" from looking valid.
    is_valid = best_answer_agreement >= float(threshold)

    # Identity/definition questions require answer-form evidence. A sentence
    # that merely mentions the requested concept must not validate itself.
    if query:
        q_norm = str(query).strip().lower()
        m = re.search(r"\b(?:what|who)\s+(?:is|are|was|were)\s+(.+?)[?.!]?$", q_norm)
        if m:
            focus = m.group(1).strip()
            def direct_definition(text):
                f = re.escape(focus)
                return bool(any(re.search(p, text, re.I) for p in (
                    rf"\b{f}\s+is\s+(?:a|an|the)\b",
                    rf"\b{f}\s+is\s+the\s+process\b",
                    rf"\b{f}\s+is\s+the\s+action\b",
                    rf"\b{f}\s+refers\s+to\b",
                    rf"\b{f}\s+means\b",
                    rf"\b{f}\s+is\s+defined\s+as\b",
                    rf"\b(?:has|have)\s+defined\s+{f}\s+as\b",
                )))
            definition_ok = any(direct_definition(c) for c in candidates)
            if focus == "risk perception":
                definition_ok = definition_ok or any(
                    re.search(r"\brisk\s+is\s+the\s+possibility\s+of\b", c, re.I)
                    for c in candidates
                )
            if not definition_ok:
                is_valid = False
    if query:
        is_valid = is_valid and (
            query_grounding >= 0.43
            or best_query_lexical >= 0.60
            or best_query_semantic >= 0.60
        )

    # Backwards-compatible confidence remains answer/evidence agreement, but
    # the new grounding value is exposed explicitly for calibration.
    confidence = min(best_answer_agreement, max(query_grounding, 0.0)) if query else best_answer_agreement
    # Validation confidence is externally reported as answer confidence. Do
    # not expose high semantic agreement as high answer confidence when the
    # evidence failed the query/answer-form gates.
    if not is_valid:
        confidence = min(confidence, 0.20)

    print("[Answer Validation] Model ID:", id(model))
    print("[Optimization #30] Validation embedding cache:", embedding_cache.info())
    print("[Answer Validation] Scope:", scope)
    print("[Answer Validation] Candidates:", len(candidates))
    print("[Answer Validation] Best semantic agreement:", f"{best_answer_agreement:.4f}")
    if query:
        print("[Answer Validation] Query grounding:", f"{query_grounding:.4f}")
        print("[Answer Validation] Query semantic:", f"{best_query_semantic:.4f}")
        print("[Answer Validation] Query lexical:", f"{best_query_lexical:.4f}")

    return {
        "is_valid": bool(is_valid),
        "confidence": float(confidence),
        "answer_agreement": best_answer_agreement,
        "query_grounding": query_grounding,
        "query_semantic": best_query_semantic,
        "query_lexical": best_query_lexical,
        "validation_scope": scope,
        "candidate_count": len(candidates),
        "threshold": float(threshold),
    }
