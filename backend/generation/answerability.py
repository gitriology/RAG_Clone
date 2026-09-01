"""Answerability gate for extractive RAG.

Retrieval answers two different questions:
1. What material is semantically close to the query?
2. Does the material actually contain enough information to answer it?

This module handles the second question.  It deliberately favors abstention
when there is no strong query/entity/reference anchor, preventing semantically
related but irrelevant chunks from becoming confident answers.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from backend.retrieval.query_matching import (
    entity_alignment,
    extract_reference,
    query_anchor_score,
)


def _clamp(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def assess_selected_evidence(
    query: str,
    evidence: Iterable[Dict[str, Any]] | None,
) -> Dict[str, Any]:
    items = [x for x in (evidence or []) if isinstance(x, dict)]

    if not items:
        return {
            "answerable": False,
            "score": 0.0,
            "reason": "no_selected_evidence",
            "best_anchor": 0.0,
            "best_target": 0.0,
            "best_semantic": 0.0,
            "best_quality": 0.0,
            "reference_query": extract_reference(query),
        }

    best_anchor = 0.0
    best_target = 0.0
    best_semantic = 0.0
    best_quality = 0.0
    best_relevance = 0.0
    best_item = None

    for item in items:
        text = str(item.get("text", "")).strip()
        anchor = max(
            _clamp(item.get("entity_score", 0.0)),
            query_anchor_score(query, text),
        )
        target = _clamp(item.get("target_score", 0.0))
        semantic = _clamp(item.get("semantic", item.get("similarity", 0.0)))
        quality = _clamp(item.get("quality", 0.0))
        relevance = _clamp(item.get("relevance", item.get("evidence_score", 0.0)))

        candidate = (
            0.35 * anchor
            + 0.30 * target
            + 0.15 * semantic
            + 0.10 * quality
            + 0.10 * relevance
        )

        if candidate > (0.35 * best_anchor + 0.30 * best_target +
                         0.15 * best_semantic + 0.10 * best_quality +
                         0.10 * best_relevance):
            best_item = item

        best_anchor = max(best_anchor, anchor)
        best_target = max(best_target, target)
        best_semantic = max(best_semantic, semantic)
        best_quality = max(best_quality, quality)
        best_relevance = max(best_relevance, relevance)

    score = (
        0.35 * best_anchor
        + 0.30 * best_target
        + 0.15 * best_semantic
        + 0.10 * best_quality
        + 0.10 * best_relevance
    )

    reference = extract_reference(query)

    # Strict gates.  These are intentionally asymmetric: false positives are
    # more damaging than asking the user to reformulate an unsupported query.
    if reference:
        answerable = (
            best_anchor >= 0.95
            and best_target >= 0.45
            and best_quality >= 0.60
            and best_semantic >= 0.55
        )
    else:
        answerable = (
            best_anchor >= 0.80
            and best_target >= 0.40
            and best_quality >= 0.55
            and best_semantic >= 0.50
        )

    if answerable:
        reason = "strong_answer_bearing_evidence"
    elif best_anchor < 0.50:
        reason = "missing_query_entity_or_reference_anchor"
    elif best_target < 0.40:
        reason = "evidence_does_not_address_question_target"
    elif best_quality < 0.55:
        reason = "evidence_quality_too_low"
    else:
        reason = "evidence_semantic_support_too_weak"

    return {
        "answerable": bool(answerable),
        "score": _clamp(score),
        "reason": reason,
        "best_anchor": _clamp(best_anchor),
        "best_target": _clamp(best_target),
        "best_semantic": _clamp(best_semantic),
        "best_quality": _clamp(best_quality),
        "best_relevance": _clamp(best_relevance),
        "reference_query": reference,
        "best_evidence": best_item.get("text", "") if best_item else "",
    }
