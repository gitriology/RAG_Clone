from backend.reranker.scoring.confidence.confidence_score import (
    build_confidence_breakdown,
    compute_confidence,
)


def test_strong_direct_evidence_is_high_but_not_saturated():
    documents = [
        {"rerank_score": 4.0, "context_score": 0.80, "source": "vaccination.pdf"},
        {"rerank_score": 2.0, "context_score": 0.30, "source": "vaccination.pdf"},
        {"rerank_score": 1.0, "context_score": 0.20, "source": "other.pdf"},
    ]
    evidence = [{
        "similarity": 0.84,
        "relevance": 0.92,
        "quality": 1.0,
        "contamination": 0.0,
        "targets": ["identity"],
        "source": "vaccination.pdf",
    }]
    validation = {"is_valid": True, "confidence": 1.0}

    score = compute_confidence(documents, validation, 0.75, evidence=evidence, answer="x")
    assert 0.70 <= score < 0.95


def test_no_evidence_is_capped():
    score = compute_confidence(
        [{"rerank_score": 4.0, "context_score": 0.9}],
        {"is_valid": True, "confidence": 1.0},
        0.95,
        evidence=[],
        answer="unsupported",
    )
    assert score <= 0.45


def test_breakdown_matches_confidence():
    docs = [{"rerank_score": 3.0, "context_score": 0.8, "source": "a.pdf"}]
    evidence = [{
        "similarity": 0.9,
        "relevance": 0.9,
        "quality": 1.0,
        "contamination": 0.0,
        "targets": ["identity"],
        "source": "a.pdf",
    }]
    validation = {"is_valid": True, "confidence": 0.95}

    score = compute_confidence(docs, validation, 0.8, evidence=evidence, answer="x")
    breakdown = build_confidence_breakdown(docs, validation, 0.8, evidence=evidence, answer="x")
    assert abs(score - breakdown["calibrated_confidence"]) < 1e-9
