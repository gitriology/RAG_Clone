"""Tests for Optimization #30: context-validation embedding reuse."""

from __future__ import annotations

import numpy as np
import pytest

from backend.generation.validation.embedding_cache import ValidationEmbeddingCache



class FakeValidationModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        values = [str(x) for x in texts]
        self.calls.append(values)
        vectors = []
        for text in values:
            # Deterministic, non-zero vector derived from text length/content.
            base = float(sum(ord(c) for c in text) % 997 + 1)
            vectors.append([base, base + 1.0, base + 2.0, base + 3.0])
        return np.asarray(vectors, dtype=np.float32)


def test_validation_cache_batches_misses_and_reuses_hits():
    model = FakeValidationModel()
    cache = ValidationEmbeddingCache(model=model)

    first = cache.encode(["alpha", "beta", "alpha"])
    second = cache.encode(["beta", "gamma", "alpha"])

    assert first.shape == (3, 4)
    assert second.shape == (3, 4)
    assert len(model.calls) == 2
    assert model.calls[0] == ["alpha", "beta"]
    assert model.calls[1] == ["gamma"]
    assert cache.info() == {
        "cache_size": 3,
        "encode_calls": 2,
        "encoded_items": 3,
        "reused_items": 2,
    }


def test_context_and_answer_validation_share_query_and_sentence_embeddings(monkeypatch):
    pytest.importorskip("sentence_transformers")
    from backend.generation.guards.answer_validator import validate_answer
    from backend.generation.validation.context_validator import validate_context

    model = FakeValidationModel()
    cache = ValidationEmbeddingCache(model=model)

    documents = [{
        "doc_id": "D1",
        "text": "FAISS is a library for efficient similarity search.",
    }]

    validate_context(
        "What is FAISS?",
        documents,
        embedding_cache=cache,
    )

    calls_after_context = len(model.calls)
    assert calls_after_context == 2  # query + shared document/sentence embedding

    result = validate_answer(
        "FAISS is a library for efficient similarity search.",
        documents,
        evidence=[{"text": "FAISS is a library for efficient similarity search."}],
        query="What is FAISS?",
        embedding_cache=cache,
    )

    assert result["candidate_count"] == 1
    assert result["validation_method"] == "similarity_based_answer_support"

    # The answer text is identical to the cached document/evidence text, and
    # the query was already embedded during context validation. Therefore no
    # additional model.encode() call should occur.
    assert len(model.calls) == calls_after_context
    assert len(model.calls) == 2


def test_context_validation_preserves_backward_compatible_call():
    pytest.importorskip("sentence_transformers")
    from backend.generation.validation.context_validator import validate_context

    model = FakeValidationModel()
    cache = ValidationEmbeddingCache(model=model)
    documents = [{"doc_id": "D1", "text": "This is valid evidence for testing."}]

    result = validate_context(
        "What is testing?",
        documents,
        embedding_cache=cache,
    )

    assert result is documents
    assert "context_score" in documents[0]
