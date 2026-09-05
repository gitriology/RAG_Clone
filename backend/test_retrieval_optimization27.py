"""Tests for Optimization #27 retrieval-layer improvements."""

import numpy as np
import pytest


def test_shared_bm25_tokenizer_handles_case_and_punctuation():
    bm25 = pytest.importorskip("backend.retrieval.lexical.bm25")
    assert bm25.tokenize("Hello, WORLD! Article-12.") == [
        "hello", "world", "article", "12"
    ]


def test_embedding_dimension_is_exposed(monkeypatch):
    pytest.importorskip("sentence_transformers")
    module = __import__(
        "backend.retrieval.dense.embedder",
        fromlist=["get_embedding_dimension", "ModelRegistry"],
    )

    class FakeModel:
        def get_sentence_embedding_dimension(self):
            return 768

    monkeypatch.setattr(
        module.ModelRegistry,
        "get_embedding_model",
        classmethod(lambda cls: FakeModel()),
    )
    assert module.get_embedding_dimension() == 768


def test_hybrid_query_cache_reuses_completed_result(monkeypatch):
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("rank_bm25")
    module = __import__(
        "backend.retrieval.hybrid.hybrid_search",
        fromlist=["hybrid_search", "clear_query_cache", "query_cache_info"],
    )

    class FakeIndex:
        def search(self, embedding, candidate_k):
            scores = np.array([[0.9, 0.5]], dtype=np.float32)
            ids = np.array([[0, 1]], dtype=np.int64)
            return scores, ids

    class FakeBM25:
        def get_scores(self, tokens):
            return np.array([2.0, 1.0], dtype=np.float32)

    calls = {"encode": 0, "faiss": 0, "bm25": 0}

    def fake_encode_query(query):
        calls["encode"] += 1
        return np.ones((1, 3), dtype=np.float32)

    original_search = FakeIndex.search
    def counted_search(self, embedding, candidate_k):
        calls["faiss"] += 1
        return original_search(self, embedding, candidate_k)

    original_scores = FakeBM25.get_scores
    def counted_scores(self, tokens):
        calls["bm25"] += 1
        return original_scores(self, tokens)

    monkeypatch.setattr(module, "encode_query", fake_encode_query)
    FakeIndex.search = counted_search
    FakeBM25.get_scores = counted_scores
    module.clear_query_cache()

    index = FakeIndex()
    bm25 = FakeBM25()
    texts = ["alpha", "beta"]

    first = module.hybrid_search("alpha", index, bm25, texts, k=2, candidate_k=2)
    second = module.hybrid_search("alpha", index, bm25, texts, k=2, candidate_k=2)

    assert first == second
    assert calls == {"encode": 1, "faiss": 1, "bm25": 1}
    assert module.query_cache_info()["size"] == 1
