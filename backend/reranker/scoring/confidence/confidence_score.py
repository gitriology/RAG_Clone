"""
Evidence-aware confidence estimation for the production RAG pipeline.

Important distinction
---------------------
CrossEncoder scores, cosine similarities and retrieval scores are model
signals.  They are NOT probabilities.  This module therefore combines them
as bounded evidence signals and applies conservative gates instead of
pretending that a sigmoid of a ranking logit is a calibrated probability.

The returned ``pipeline_confidence`` is a confidence *estimate*.  It becomes
a statistically calibrated probability only after fitting calibration
parameters on a labelled QA validation set.  The implementation exposes the
raw components and calibration metadata so that such a fit can be added
without changing the API contract.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional


CALIBRATION_VERSION = "evidence_aware_v4_answerability"


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return low
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


def _get(doc: Any, key: str, default: Any = 0.0) -> Any:
    if isinstance(doc, dict):
        return doc.get(key, default)
    return getattr(doc, key, default)


def _source(doc: Any) -> str:
    if isinstance(doc, dict):
        value = doc.get("source", "")
        if value:
            return str(value).strip()
        metadata = doc.get("metadata", {})
        if isinstance(metadata, dict):
            return str(metadata.get("source", "")).strip()
        return ""

    metadata = getattr(doc, "metadata", {})
    if isinstance(metadata, dict):
        return str(metadata.get("source", "")).strip()
    return ""


def _ranking_confidence(documents: List[Any]) -> float:
    """Return a stable rank-separation signal in [0, 1].

    The CrossEncoder is a ranker.  Its absolute logits are intentionally not
    converted to probabilities.  We use only the separation between the best
    result and the rest of the current candidate set.
    """
    scores = []
    for doc in documents:
        value = _get(doc, "rerank_score", None)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            scores.append(value)

    if not scores:
        return 0.0

    scores.sort(reverse=True)

    if len(scores) == 1:
        return 1.0

    # Robust query-local separation.  The denominator is based on the
    # observed score spread, not the absolute logit magnitude.
    top = scores[0]
    second = scores[1]
    spread = max(1e-6, scores[0] - scores[-1])
    gap_signal = (top - second) / spread

    # Also reward a top score that is materially above the candidate mean.
    mean = sum(scores) / len(scores)
    top_vs_mean = (top - mean) / spread

    return _clamp(0.65 * gap_signal + 0.35 * top_vs_mean)


def _context_confidence(documents: List[Any]) -> float:
    """Document-level context support.

    This is deliberately secondary when sentence-level evidence exists.
    Whole-document embeddings are often diluted by unrelated PDF content.
    """
    if not documents:
        return 0.0

    values = []
    for doc in documents:
        values.append(_clamp(_get(doc, "context_score", 0.0)))

    values.sort(reverse=True)
    best = values[0]
    mean = sum(values) / len(values)

    return _clamp(0.75 * best + 0.25 * mean)


def _answer_agreement(validation: Dict[str, Any]) -> float:
    """Semantic agreement with selected evidence.

    This remains a similarity signal.  A value of 1.0 means the validator
    found perfect semantic agreement, not that the answer has 100% factual
    probability.
    """
    return _clamp(validation.get("confidence", 0.0))


def _query_grounding(validation: Dict[str, Any]) -> float:
    """Independent query -> selected-evidence grounding signal.

    Backward compatible with older validation dictionaries that did not expose
    query grounding: a valid legacy result is treated as fully grounded.
    """
    if "query_grounding" in validation:
        return _clamp(validation.get("query_grounding", 0.0))
    return 1.0 if bool(validation.get("is_valid", False)) else 0.0


def _answerable(validation: Dict[str, Any]) -> bool:
    """Read answerability without breaking legacy validation callers."""
    if "answerable" in validation:
        return bool(validation.get("answerable", False))
    return bool(validation.get("is_valid", False))


def _selected_evidence_stats(
    evidence: Optional[Iterable[Dict[str, Any]]],
) -> Dict[str, float]:
    items = [x for x in (evidence or []) if isinstance(x, dict)]

    if not items:
        return {
            "count": 0.0,
            "relevance": 0.0,
            "semantic": 0.0,
            "quality": 0.0,
            "contamination": 1.0,
            "target": 0.0,
            "source_consistency": 0.0,
        }

    def weighted(values: List[float]) -> float:
        if not values:
            return 0.0
        values = sorted(values, reverse=True)
        best = values[0]
        mean = sum(values) / len(values)
        return _clamp(0.70 * best + 0.30 * mean)

    relevance = weighted([
        _clamp(x.get("relevance", x.get("evidence_score", 0.0)))
        for x in items
    ])
    semantic = weighted([
        _clamp(x.get("similarity", x.get("semantic", 0.0)))
        for x in items
    ])
    quality = weighted([
        _clamp(x.get("quality", 1.0))
        for x in items
    ])
    contamination = max([
        _clamp(x.get("contamination", 0.0))
        for x in items
    ])

    target_values = [
        1.0 if x.get("targets") else 0.0
        for x in items
    ]
    target = _clamp(sum(target_values) / len(target_values))

    sources = [
        str(x.get("source", "")).strip()
        for x in items
        if str(x.get("source", "")).strip()
    ]
    if not sources:
        source_consistency = 1.0
    else:
        counts: Dict[str, int] = {}
        for value in sources:
            counts[value] = counts.get(value, 0) + 1
        source_consistency = _clamp(max(counts.values()) / len(sources))

    return {
        "count": float(len(items)),
        "relevance": relevance,
        "semantic": semantic,
        "quality": quality,
        "contamination": contamination,
        "target": target,
        "source_consistency": source_consistency,
    }


def _selected_source_consistency(
    documents: List[Any],
    evidence_stats: Dict[str, float],
) -> float:
    """Use evidence provenance when available; otherwise fall back safely."""
    if evidence_stats["count"] > 0 and evidence_stats["source_consistency"] > 0:
        return evidence_stats["source_consistency"]

    sources = [_source(doc) for doc in documents if _source(doc)]
    if not sources:
        return 0.5

    counts: Dict[str, int] = {}
    for value in sources:
        counts[value] = counts.get(value, 0) + 1
    return _clamp(max(counts.values()) / len(sources))


def _calibrate(raw_score: float) -> float:
    """Conservative deterministic calibration curve.

    Until a labelled calibration set is fitted, do not use a steep sigmoid:
    it turns a score such as .80 into .82 and creates false certainty.
    Instead, gently shrink scores toward the centre of the interval.
    """
    raw_score = _clamp(raw_score)
    return _clamp(0.08 + 0.88 * raw_score)


def compute_confidence(
    documents,
    validation,
    retrieval_confidence,
    evidence: Optional[Iterable[Dict[str, Any]]] = None,
    answer: Optional[str] = None,
):
    """Compute the single confidence value used by CLI and API.

    The same function is called from ``run_pipeline``.  The API simply
    serializes the resulting value; it never recalculates or rescales it.
    """
    documents = list(documents or [])
    validation = validation if isinstance(validation, dict) else {}

    retrieval = _clamp(retrieval_confidence)
    ranking = _ranking_confidence(documents)
    context = _context_confidence(documents)
    answer_agreement = _answer_agreement(validation)
    query_grounding = _query_grounding(validation)
    answerable = _answerable(validation)

    evidence_stats = _selected_evidence_stats(evidence)
    evidence_count = evidence_stats["count"]

    if evidence_count > 0:
        relevance = evidence_stats["relevance"]
        semantic = evidence_stats["semantic"]
        quality = evidence_stats["quality"]
        contamination = evidence_stats["contamination"]
        target_support = evidence_stats["target"]
    else:
        relevance = 0.0
        semantic = 0.0
        quality = 0.0
        contamination = 1.0
        target_support = 0.0

    source_consistency = _selected_source_consistency(
        documents,
        evidence_stats,
    )

    # Direct evidence is intentionally dominant.  Retrieval/ranking are
    # supporting signals and must not overpower a clean selected sentence.
    raw = (
        0.12 * retrieval
        + 0.08 * ranking
        + 0.08 * context
        + 0.12 * answer_agreement
        + 0.18 * query_grounding
        + 0.22 * relevance
        + 0.10 * semantic
        + 0.06 * quality
        + 0.06 * target_support
        + 0.08 * source_consistency
    )

    # Contamination is a real negative signal.  It is deliberately based on
    # selected evidence rather than unrelated retrieved documents.
    raw -= 0.16 * contamination

    # Hard safety gates.  These prevent a high retrieval score from producing
    # a high answer confidence when the answer is not actually supported.
    is_valid = bool(validation.get("is_valid", False))

    if evidence_count == 0:
        raw *= 0.55

    if not is_valid:
        raw *= 0.55

    # Independent answerability is a hard production safety signal.
    # Semantic agreement alone must never create high confidence.
    if not answerable:
        raw *= 0.50

    if query_grounding < 0.52:
        raw *= 0.65

    if evidence_count > 0 and target_support >= 0.5 and relevance >= 0.75:
        raw += 0.03

    raw = _clamp(raw)
    calibrated = _calibrate(raw)

    if evidence_count == 0:
        calibrated = min(calibrated, 0.45)

    if not is_valid:
        calibrated = min(calibrated, 0.40)

    if not answerable:
        calibrated = min(calibrated, 0.25)
    elif query_grounding < 0.52:
        calibrated = min(calibrated, 0.35)

    calibrated = _clamp(calibrated)

    breakdown = {
        "method": CALIBRATION_VERSION,
        "raw_composite": raw,
        "calibrated_confidence": calibrated,
        "statistically_fitted": False,
        "calibration_note": (
            "Deterministic evidence-aware calibration. "
            "Fit on labelled QA outcomes for probabilistic calibration."
        ),
        "components": {
            "retrieval_confidence": retrieval,
            "ranking_concentration": ranking,
            "context_support": context,
            "answer_agreement": answer_agreement,
            "query_grounding": query_grounding,
            "answerable": answerable,
            "evidence_relevance": relevance,
            "evidence_semantic": semantic,
            "evidence_quality": quality,
            "target_support": target_support,
            "source_consistency": source_consistency,
            "evidence_contamination": contamination,
        },
        "selected_evidence_count": int(evidence_count),
        "answer_valid": is_valid,
        "answerable": answerable,
        "query_grounding": query_grounding,
    }

    print("\n" + "=" * 60)
    print("Pipeline Confidence Debug")
    print("=" * 60)
    print(f"Retrieval Confidence     : {retrieval:.4f}")
    print(f"Ranking Concentration    : {ranking:.4f}")
    print(f"Context Support          : {context:.4f}")
    print(f"Answer Agreement        : {answer_agreement:.4f}")
    print(f"Query Grounding         : {query_grounding:.4f}")
    print(f"Answerable              : {answerable}")
    print(f"Evidence Relevance       : {relevance:.4f}")
    print(f"Evidence Semantic        : {semantic:.4f}")
    print(f"Evidence Quality         : {quality:.4f}")
    print(f"Target Support           : {target_support:.4f}")
    print(f"Source Consistency       : {source_consistency:.4f}")
    print(f"Evidence Contamination   : {contamination:.4f}")
    print(f"Raw Composite            : {raw:.4f}")
    print("-" * 60)
    print(f"Final Pipeline Confidence: {calibrated:.4f}")
    print("=" * 60)

    # ``compute_confidence`` historically returned a float.  Keep that API
    # stable; run_pipeline exposes the detailed breakdown separately.
    return calibrated


def build_confidence_breakdown(
    documents,
    validation,
    retrieval_confidence,
    evidence: Optional[Iterable[Dict[str, Any]]] = None,
    answer: Optional[str] = None,
) -> Dict[str, Any]:
    """Return diagnostics without duplicating confidence calculations.

    The production pipeline calls this after ``compute_confidence`` only for
    presentation.  The function uses the same formula and is kept as a
    separate diagnostic API for tests and observability.
    """
    documents = list(documents or [])
    validation = validation if isinstance(validation, dict) else {}
    evidence_stats = _selected_evidence_stats(evidence)

    retrieval = _clamp(retrieval_confidence)
    ranking = _ranking_confidence(documents)
    context = _context_confidence(documents)
    answer_agreement = _answer_agreement(validation)
    query_grounding = _query_grounding(validation)
    answerable = _answerable(validation)

    relevance = evidence_stats["relevance"] if evidence_stats["count"] else 0.0
    semantic = evidence_stats["semantic"] if evidence_stats["count"] else 0.0
    quality = evidence_stats["quality"] if evidence_stats["count"] else 0.0
    contamination = evidence_stats["contamination"] if evidence_stats["count"] else 1.0
    target_support = evidence_stats["target"] if evidence_stats["count"] else 0.0
    source_consistency = _selected_source_consistency(documents, evidence_stats)

    raw = (
        0.12 * retrieval
        + 0.08 * ranking
        + 0.08 * context
        + 0.12 * answer_agreement
        + 0.18 * query_grounding
        + 0.22 * relevance
        + 0.10 * semantic
        + 0.06 * quality
        + 0.06 * target_support
        + 0.08 * source_consistency
    )
    raw -= 0.16 * contamination

    if evidence_stats["count"] == 0:
        raw *= 0.55
    if not bool(validation.get("is_valid", False)):
        raw *= 0.55
    if not answerable:
        raw *= 0.50
    if query_grounding < 0.52:
        raw *= 0.65
    if evidence_stats["count"] > 0 and target_support >= 0.5 and relevance >= 0.75:
        raw += 0.03

    raw = _clamp(raw)
    calibrated = _calibrate(raw)
    if evidence_stats["count"] == 0:
        calibrated = min(calibrated, 0.45)
    if not bool(validation.get("is_valid", False)):
        calibrated = min(calibrated, 0.40)
    if not answerable:
        calibrated = min(calibrated, 0.25)
    elif query_grounding < 0.52:
        calibrated = min(calibrated, 0.35)

    return {
        "method": CALIBRATION_VERSION,
        "raw_composite": raw,
        "calibrated_confidence": _clamp(calibrated),
        "statistically_fitted": False,
        "components": {
            "retrieval_confidence": retrieval,
            "ranking_concentration": ranking,
            "context_support": context,
            "answer_agreement": answer_agreement,
            "query_grounding": query_grounding,
            "answerable": answerable,
            "evidence_relevance": relevance,
            "evidence_semantic": semantic,
            "evidence_quality": quality,
            "target_support": target_support,
            "source_consistency": source_consistency,
            "evidence_contamination": contamination,
        },
        "selected_evidence_count": int(evidence_stats["count"]),
        "answer_valid": bool(validation.get("is_valid", False)),
        "answerable": answerable,
        "query_grounding": query_grounding,
    }
