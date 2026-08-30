"""
Confidence calibration for the production RAG pipeline.

The previous implementation averaged CrossEncoder sigmoid(logit) values.
Those logits are ranking scores, not calibrated probabilities, so a query can
receive a misleadingly low confidence even when the selected evidence is
strong.

This module deliberately separates:
    1. retrieval confidence
    2. ranking concentration
    3. contextual support
    4. answer/evidence agreement
    5. evidence quality
    6. contamination / disagreement

The final value is a calibrated *pipeline confidence estimate*, not a
probability of factual truth unless this module is fitted against labelled
validation data. The score is bounded to [0, 1] and is stable across
documents with different CrossEncoder logit scales.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return low
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


def _sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def _softmax(values: Iterable[float], temperature: float = 1.0) -> List[float]:
    values = [float(v) for v in values]
    if not values:
        return []

    temperature = max(float(temperature), 1e-6)
    scaled = [v / temperature for v in values]
    maximum = max(scaled)
    exps = [math.exp(max(-60.0, min(60.0, v - maximum))) for v in scaled]
    total = sum(exps)

    if total <= 0.0:
        return [1.0 / len(values)] * len(values)

    return [v / total for v in exps]


def _get(doc: Any, key: str, default: float = 0.0) -> float:
    if isinstance(doc, dict):
        return _clamp(doc.get(key, default))
    return _clamp(getattr(doc, key, default))


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
        }

    relevance = [_clamp(x.get("relevance", x.get("evidence_score", 0.0))) for x in items]
    semantic = [_clamp(x.get("similarity", x.get("semantic", 0.0))) for x in items]
    quality = [_clamp(x.get("quality", 1.0)) for x in items]
    contamination = [_clamp(x.get("contamination", 0.0)) for x in items]

    targets = [
        1.0 if x.get("targets") else 0.0
        for x in items
    ]

    return {
        "count": float(len(items)),
        "relevance": sum(relevance) / len(relevance),
        "semantic": sum(semantic) / len(semantic),
        "quality": sum(quality) / len(quality),
        "contamination": sum(contamination) / len(contamination),
        "target": sum(targets) / len(targets),
    }


def _ranking_confidence(documents: List[Any]) -> float:
    """
    Convert ranking logits into a query-local concentration score.

    We do NOT call sigmoid(logit) a probability. The CrossEncoder is used
    for ranking, so confidence comes from how strongly the best documents
    separate from the rest of the retrieved set.
    """
    logits = [
        float(
            doc.get("rerank_score", 0.0)
            if isinstance(doc, dict)
            else getattr(doc, "rerank_score", 0.0)
        )
        for doc in documents
    ]

    if not logits:
        return 0.0

    probs = _softmax(logits, temperature=1.0)
    top = max(probs)

    if len(probs) == 1:
        return top

    # Normalise top probability so uniform ranking = 0 and perfect
    # concentration = 1.
    uniform = 1.0 / len(probs)
    concentration = (top - uniform) / max(1e-6, 1.0 - uniform)

    # Also consider the gap between the best and second-best logit.
    ordered = sorted(logits, reverse=True)
    gap = ordered[0] - ordered[1]
    gap_score = _sigmoid(gap)

    return _clamp(0.65 * concentration + 0.35 * gap_score)


def _context_confidence(documents: List[Any]) -> float:
    if not documents:
        return 0.0

    context = [_get(doc, "context_score") for doc in documents]
    best = max(context)
    average = sum(context) / len(context)

    # The best context is more important than the average because long
    # documents often contain unrelated material.
    return _clamp(0.70 * best + 0.30 * average)


def _answer_confidence(validation: Dict[str, Any]) -> float:
    """
    Treat answer validation as semantic agreement, not as a probability.

    Cosine similarity is shifted into a conservative confidence band:
        <= .30 -> 0
        .30-.85 -> 0-.95
        >= .85 -> 1
    """
    raw = _clamp(validation.get("confidence", 0.0))

    if raw <= 0.30:
        return 0.0

    scaled = (raw - 0.30) / 0.55
    return _clamp(scaled)


def _source_consistency(documents: List[Any]) -> float:
    if not documents:
        return 0.0

    sources = []
    for doc in documents:
        source = (
            doc.get("source", "")
            if isinstance(doc, dict)
            else getattr(doc, "metadata", {}).get("source", "")
        )
        source = str(source).strip()
        if source:
            sources.append(source)

    if not sources:
        return 0.5

    counts: Dict[str, int] = {}
    for source in sources:
        counts[source] = counts.get(source, 0) + 1

    dominant = max(counts.values())
    return _clamp(dominant / len(sources))


def _calibrate_composite(raw_score: float) -> float:
    """
    Smooth the composite score around the useful middle of the scale.

    This is a deterministic calibration curve. It should not be described
    as statistically calibrated until fitted against labelled evaluation
    data.
    """
    raw_score = _clamp(raw_score)

    # Keep strong evidence strong, but avoid saturating too early.
    calibrated = _sigmoid(5.0 * (raw_score - 0.50))
    return _clamp(calibrated)


def compute_confidence(
    documents,
    validation,
    retrieval_confidence,
    evidence: Optional[Iterable[Dict[str, Any]]] = None,
    answer: Optional[str] = None,
):
    """
    Compute one confidence value shared by CLI and API.

    Important:
        The same function is called by run_rerank and therefore by the
        FastAPI endpoint. The frontend must display this value unchanged.

    Returns a number in [0, 1].
    """
    documents = list(documents or [])
    validation = validation if isinstance(validation, dict) else {}

    retrieval = _clamp(retrieval_confidence)
    ranking = _ranking_confidence(documents)
    context = _context_confidence(documents)
    answer_score = _answer_confidence(validation)
    source_consistency = _source_consistency(documents)

    evidence_stats = _selected_evidence_stats(evidence)

    if evidence_stats["count"] > 0:
        evidence_relevance = evidence_stats["relevance"]
        evidence_semantic = evidence_stats["semantic"]
        evidence_quality = evidence_stats["quality"]
        contamination = evidence_stats["contamination"]
        target_support = evidence_stats["target"]
    else:
        evidence_relevance = 0.0
        evidence_semantic = 0.0
        evidence_quality = 0.0
        contamination = 1.0
        target_support = 0.0

    # Evidence-aware answer grounding. When actual selected evidence exists,
    # it is stronger evidence of answer quality than whole-document averages.
    grounding = _clamp(
        0.45 * evidence_relevance
        + 0.25 * evidence_semantic
        + 0.20 * answer_score
        + 0.10 * evidence_quality
    )

    # Base confidence. Retrieval remains important, but it no longer
    # dominates the final score.
    raw = (
        0.20 * retrieval
        + 0.15 * ranking
        + 0.15 * context
        + 0.25 * grounding
        + 0.10 * source_consistency
        + 0.10 * target_support
        + 0.05 * answer_score
    )

    # Penalise malformed evidence and cross-domain/source contamination.
    raw -= 0.15 * contamination

    # If the answer is valid and has direct selected evidence, avoid
    # treating irrelevant retrieved documents as evidence against it.
    if (
        bool(validation.get("is_valid", False))
        and evidence_stats["count"] > 0
        and target_support >= 0.5
        and evidence_relevance >= 0.75
    ):
        raw += 0.05

    overall = _calibrate_composite(raw)

    # A valid answer cannot receive a high confidence score with no
    # supporting evidence.
    if evidence_stats["count"] == 0:
        overall = min(overall, 0.45)

    if not bool(validation.get("is_valid", False)):
        overall = min(overall, 0.40)

    overall = _clamp(overall)

    print("\n" + "=" * 60)
    print("Pipeline Confidence Debug")
    print("=" * 60)
    print(f"Retrieval Confidence     : {retrieval:.4f}")
    print(f"Ranking Concentration    : {ranking:.4f}")
    print(f"Context Support          : {context:.4f}")
    print(f"Answer Validation        : {answer_score:.4f}")
    print(f"Evidence Relevance       : {evidence_relevance:.4f}")
    print(f"Evidence Semantic        : {evidence_semantic:.4f}")
    print(f"Evidence Quality         : {evidence_quality:.4f}")
    print(f"Target Support           : {target_support:.4f}")
    print(f"Source Consistency       : {source_consistency:.4f}")
    print(f"Evidence Contamination   : {contamination:.4f}")
    print(f"Raw Composite            : {_clamp(raw):.4f}")
    print("-" * 60)
    print(f"Final Pipeline Confidence: {overall:.4f}")
    print("=" * 60)

    return overall
