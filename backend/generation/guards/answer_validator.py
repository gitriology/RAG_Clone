"""Answer validation for the production extractive RAG pipeline.

Validation must answer a different question from semantic similarity:

    Does the selected evidence actually answer the user's query?

An answer agreeing with itself is not evidence of correctness.  Therefore
this validator combines semantic answer/evidence agreement with independent
query-to-evidence answerability checks from ``query_matching``.
"""

from __future__ import annotations

from typing import Iterable, Optional

from sklearn.metrics.pairwise import cosine_similarity

from backend.models.model_registry import ModelRegistry
from backend.retrieval.query_matching import assess_answerability


def _clamp(value, low=0.0, high=1.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, value))


def validate_answer(
    answer,
    documents,
    threshold=0.40,
    evidence: Optional[Iterable] = None,
    query: str = "",
):
    """Validate an answer against selected evidence and the original query.

    ``evidence`` is authoritative.  We intentionally do not fall back to
    arbitrary retrieved documents when selected evidence is empty because
    that can turn an unsupported answer into a seemingly valid one.

    Backward compatibility:
    - ``query`` is optional.
    - existing callers that only rely on semantic validation still receive
      ``confidence`` and ``answer_agreement``.
    """
    if not answer or not str(answer).strip():
        return {
            "is_valid": False,
            "confidence": 0.0,
            "answer_agreement": 0.0,
            "validation_scope": "empty_answer",
            "answerable": False,
            "query_grounding": 0.0,
        }

    candidates = []

    for item in evidence or []:
        if isinstance(item, dict):
            text = item.get("text", "")
        else:
            text = str(item)

        if text and str(text).strip():
            candidates.append(str(text).strip())

    scope = "selected_evidence"

    # Do NOT use arbitrary retrieved documents as a validation fallback.
    if not candidates:
        return {
            "is_valid": False,
            "confidence": 0.0,
            "answer_agreement": 0.0,
            "validation_scope": scope,
            "candidate_count": 0,
            "answerable": False,
            "query_grounding": 0.0,
            "answerability_reason": "no_selected_evidence",
            "matched_focus": [],
            "missing_focus": [],
        }

    model = ModelRegistry.get_validation_model()

    answer_embedding = model.encode(
        [str(answer)],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    candidate_embeddings = model.encode(
        candidates,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    scores = cosine_similarity(
        answer_embedding,
        candidate_embeddings,
    )[0]

    best = float(max(scores)) if len(scores) else 0.0

    # Independent query -> evidence assessment.  This is the important
    # anti-self-validation signal.
    if query and str(query).strip():
        answerability = assess_answerability(
            str(query),
            candidates,
            answer=str(answer),
        )
        query_grounding = _clamp(answerability.get("score", 0.0))
        answerable = bool(answerability.get("answerable", False))
    else:
        # Legacy callers have no query. Preserve the previous semantic-only
        # behavior rather than unexpectedly invalidating their answers.
        answerability = {
            "answerable": best >= float(threshold),
            "score": best,
            "reason": "query_not_provided_legacy_mode",
            "matched_focus": [],
            "missing_focus": [],
        }
        query_grounding = best
        answerable = best >= float(threshold)

    # Semantic agreement is necessary but insufficient.  A selected sentence
    # that repeats the answer can score 1.0 even when it does not answer the
    # question. Query grounding is therefore an independent gate.
    semantic_ok = best >= float(threshold)

    if query and str(query).strip():
        is_valid = semantic_ok and answerable and query_grounding >= 0.52
    else:
        is_valid = semantic_ok

    print("[Answer Validation] Model ID:", id(model))
    print("[Answer Validation] Scope:", scope)
    print("[Answer Validation] Candidates:", len(candidates))
    print("[Answer Validation] Best semantic agreement:", f"{best:.4f}")

    if query and str(query).strip():
        print(
            "[Answer Validation] Query grounding:",
            f"{query_grounding:.4f}",
        )
        print(
            "[Answer Validation] Answerable:",
            answerable,
        )
        print(
            "[Answer Validation] Reason:",
            answerability.get("reason", ""),
        )

    return {
        "is_valid": bool(is_valid),
        # Backwards-compatible semantic agreement field.
        "confidence": best,
        "answer_agreement": best,
        "validation_scope": scope,
        "candidate_count": len(candidates),
        "threshold": float(threshold),
        "answerable": bool(answerable),
        "query_grounding": query_grounding,
        "answerability_reason": answerability.get("reason", ""),
        "matched_focus": list(answerability.get("matched_focus", [])),
        "missing_focus": list(answerability.get("missing_focus", [])),
        "answerability_score": _clamp(answerability.get("score", 0.0)),
    }
