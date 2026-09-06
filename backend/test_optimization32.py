"""Tests for Optimization #32: batched query + sentence embedding selection."""

from __future__ import annotations

import pytest


def test_query_and_candidate_embeddings_are_encoded_in_one_pass(monkeypatch):
    pytest.importorskip("sentence_transformers")

    import torch
    from backend.generation import answer_generator

    class FakeEmbeddingModel:
        def __init__(self):
            self.calls = []

        def encode(self, texts, **kwargs):
            values = [str(x) for x in texts]
            self.calls.append(values)
            rows = []
            for i, text in enumerate(values, start=1):
                rows.append([float(i), float(len(text) + i), 1.0, 0.5])
            return torch.tensor(rows, dtype=torch.float32)

    model = FakeEmbeddingModel()
    monkeypatch.setattr(
        answer_generator.ModelRegistry,
        "get_embedding_model",
        classmethod(lambda cls: model),
    )

    query = "What is FAISS?"
    sentences = [
        "FAISS is a library for efficient similarity search.",
        "BM25 is a lexical retrieval method.",
    ]

    query_embedding, candidate_embeddings = answer_generator._encode_query_and_candidates(
        query,
        sentences,
    )

    assert len(model.calls) == 1
    assert model.calls[0] == [query, *sentences]
    assert tuple(query_embedding.shape) == (4,)
    assert tuple(candidate_embeddings.shape) == (2, 4)


def test_generate_answer_reports_batched_query_embedding(monkeypatch):
    pytest.importorskip("sentence_transformers")

    import torch
    from backend.generation import answer_generator

    class FakeEmbeddingModel:
        def encode(self, texts, **kwargs):
            rows = []
            for text in texts:
                base = float(sum(ord(c) for c in str(text)) % 997 + 1)
                rows.append([base, base + 1, base + 2, base + 3])
            return torch.nn.functional.normalize(
                torch.tensor(rows, dtype=torch.float32), p=2, dim=1
            )

    monkeypatch.setattr(
        answer_generator.ModelRegistry,
        "get_embedding_model",
        classmethod(lambda cls: FakeEmbeddingModel()),
    )

    result = answer_generator.generate_answer(
        "What is FAISS?",
        [{
            "doc_id": "D1",
            "source": "test.pdf",
            "domain": "technical",
            "text": "FAISS is a library for efficient similarity search.",
        }],
        max_sentences=1,
    )

    assert result["embedding_passes"] == 1
    assert result["query_embedding_batched"] is True
    assert result["sentence_embeddings"]
