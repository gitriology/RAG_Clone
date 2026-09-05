"""Tests for Optimization #29: shared Cross-Encoder + no duplicate reranking."""

import pytest

from backend.ms_arc.state.retrieval_state import RetrievedDocument
from backend.reranker.model.reranker import get_shared_reranker_model, rerank


def test_reranker_registry_returns_same_instance(monkeypatch):
    pytest.importorskip("sentence_transformers")
    from backend.models.model_registry import ModelRegistry
    class FakeCrossEncoder:
        pass

    created = []

    def factory(*args, **kwargs):
        model = FakeCrossEncoder()
        created.append(model)
        return model

    monkeypatch.setattr(
        "backend.models.model_registry.CrossEncoder",
        factory,
    )
    ModelRegistry._reranker_model = None

    first = get_shared_reranker_model()
    second = get_shared_reranker_model()

    assert first is second
    assert len(created) == 1

    ModelRegistry._reranker_model = None


def test_downstream_rerank_reuses_scores_without_predict(monkeypatch):
    class ExplodingModel:
        def predict(self, *args, **kwargs):
            raise AssertionError("duplicate CrossEncoder inference occurred")

    monkeypatch.setattr(
        "backend.reranker.model.reranker.get_shared_reranker_model",
        lambda: ExplodingModel(),
    )

    docs = [
        RetrievedDocument("1", "one", rerank_score=0.20),
        RetrievedDocument("2", "two", rerank_score=0.90),
        RetrievedDocument("3", "three", rerank_score=0.50),
    ]

    ranked = rerank("query", docs)
    assert [doc.doc_id for doc in ranked] == ["2", "3", "1"]


def test_downstream_rerank_rejects_unscored_documents():
    docs = [
        {"doc_id": "1", "text": "one", "rerank_score": 0.20},
        {"doc_id": "2", "text": "two"},
    ]

    try:
        rerank("query", docs)
    except ValueError as exc:
        assert "rerank_score is missing" in str(exc)
    else:
        raise AssertionError("Expected missing-score validation error")
