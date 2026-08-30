from __future__ import annotations

from typing import Iterable, Optional

from sklearn.metrics.pairwise import cosine_similarity

from backend.models.model_registry import ModelRegistry


def validate_answer(
    answer,
    documents,
    threshold=0.40,
    evidence: Optional[Iterable] = None,
):
    """
    Validate the final answer against the evidence that actually supports it.

    Previously validation compared the answer with every retrieved document.
    That allowed an irrelevant document to lower or distort the score.

    `evidence` is preferred. It should contain the selected evidence strings or
    dictionaries with a `text` field. Documents remain a safe fallback.
    """
    if not answer:
        return {
            "is_valid": False,
            "confidence": 0.0,
            "validation_scope": "empty_answer",
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

    if not candidates:
        for doc in documents or []:
            if isinstance(doc, dict):
                text = doc.get("text", "")
            else:
                text = getattr(doc, "text", "")

            if text and str(text).strip():
                candidates.append(str(text).strip())

        scope = "retrieved_documents"

    if not candidates:
        return {
            "is_valid": False,
            "confidence": 0.0,
            "validation_scope": scope,
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

    print("[Answer Validation] Model ID:", id(model))
    print("[Answer Validation] Scope:", scope)
    print("[Answer Validation] Candidates:", len(candidates))
    print("[Answer Validation] Best semantic agreement:", f"{best:.4f}")

    is_valid = best >= float(threshold)

    return {
        "is_valid": is_valid,
        # Backwards-compatible field. This is semantic agreement, not a
        # probability of factual correctness.
        "confidence": best,
        "answer_agreement": best,
        "validation_scope": scope,
        "candidate_count": len(candidates),
        "threshold": float(threshold),
    }
