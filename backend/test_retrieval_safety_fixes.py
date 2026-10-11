"""Regression tests for retrieval correctness and adaptive metrics."""

import importlib
import sys
import types

import numpy as np


def _load_hybrid_search_with_stubs(monkeypatch):
    """Load the retrieval module without downloading/loading heavyweight ML models."""
    embedder = types.ModuleType("backend.retrieval.dense.embedder")
    embedder.encode_query = lambda query: np.ones((1, 3), dtype=np.float32)
    bm25_module = types.ModuleType("backend.retrieval.lexical.bm25")
    bm25_module.tokenize = lambda query: str(query).lower().split()
    matching = types.ModuleType("backend.retrieval.query_matching")
    matching.analyze_query = lambda query: {"focus_phrases": [], "references": []}
    matching.lexical_match_score = lambda query, text: {
        "score": 0.0, "reference_match": False,
        "exact_phrase": False, "matched": [],
    }
    matching.reference_present = lambda text, phrase: False

    monkeypatch.setitem(sys.modules, "backend.retrieval.dense.embedder", embedder)
    monkeypatch.setitem(sys.modules, "backend.retrieval.lexical.bm25", bm25_module)
    monkeypatch.setitem(sys.modules, "backend.retrieval.query_matching", matching)
    sys.modules.pop("backend.retrieval.hybrid.hybrid_search", None)
    return importlib.import_module("backend.retrieval.hybrid.hybrid_search")


def test_flat_score_normalization_has_no_artificial_signal(monkeypatch):
    module = _load_hybrid_search_with_stubs(monkeypatch)
    assert np.array_equal(module._normalize([0.0, 0.0, 0.0]), np.zeros(3))
    assert np.array_equal(module._normalize([2.0, 2.0]), np.zeros(2))


def test_minmax_normalization_keeps_relative_order(monkeypatch):
    module = _load_hybrid_search_with_stubs(monkeypatch)
    scores = module._normalize([0.2, 0.5, 0.8])
    assert np.allclose(scores, [0.0, 0.5, 1.0])


def test_msarc_can_return_full_candidate_pool_for_reranking(monkeypatch):
    module = _load_hybrid_search_with_stubs(monkeypatch)

    class FakeIndex:
        def search(self, embedding, candidate_k):
            scores = np.array([[0.9, 0.8, 0.7, 0.6, 0.5]], dtype=np.float32)
            ids = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
            return scores[:, :candidate_k], ids[:, :candidate_k]

    class FakeBM25:
        def get_scores(self, tokens):
            return np.array([5, 4, 3, 2, 1], dtype=np.float32)

    texts = ["chunk alpha", "chunk beta", "chunk gamma", "chunk delta", "chunk epsilon"]
    common = dict(
        query="test query", faiss_index=FakeIndex(), bm25=FakeBM25(),
        texts=texts, k=1, candidate_k=5,
    )
    module.clear_query_cache()
    legacy = module.hybrid_search(**common)
    expanded = module.hybrid_search(**common, return_candidates=True)
    assert len(legacy["merged_results"]) == 1
    assert len(expanded["merged_results"]) == 5
    assert expanded["candidate_k"] == 5
    assert {"query_embedding_seconds", "dense_search_seconds", "bm25_search_seconds", "candidate_fusion_seconds"}.issubset(expanded["stage_timings_seconds"])
    assert expanded["embedding_cache_status"] in {"hit", "miss"}
    assert expanded["candidate_count"] == 5


def test_margin_is_bounded_for_near_zero_and_negative_logits(monkeypatch):
    from backend.ms_arc.state.retrieval_state import RetrievalState, RetrievedDocument

    model_registry = types.ModuleType("backend.models.model_registry")

    class FakeReranker:
        def predict(self, pairs):
            return np.array([0.00001, 0.0, -0.1], dtype=float)

    class FakeRegistry:
        @classmethod
        def get_reranker_model(cls):
            return FakeReranker()

    model_registry.ModelRegistry = FakeRegistry
    monkeypatch.setitem(sys.modules, "backend.models.model_registry", model_registry)
    sys.modules.pop("backend.ms_arc.confidence.margin", None)
    margin_module = importlib.import_module("backend.ms_arc.confidence.margin")

    state = RetrievalState(query="test")
    state.selected_documents = [
        RetrievedDocument(doc_id="1", text="first"),
        RetrievedDocument(doc_id="2", text="second"),
        RetrievedDocument(doc_id="3", text="third"),
    ]
    state = margin_module.compute_margin(state)
    margin = state.signals.margin.normalized_margin
    assert 0.0 <= margin <= 1.0
    assert state.signals.margin.raw_margin >= 0.0
    assert margin < 0.001  # tiny logit gap must not look highly confident


def test_prefix_stability_does_not_penalize_expanded_candidate_pool():
    from backend.ms_arc.stability.metrics import prefix_jaccard

    previous = ["a", "b", "c", "d", "e"]
    expanded = ["a", "b", "c", "d", "e", "f", "g", "h"]
    assert prefix_jaccard(previous, expanded) == 1.0


def test_health_excludes_query_complexity_and_topk_from_quality():
    from backend.evidence_state.state.evidence_state import EvidenceState, EvidenceFeature
    from backend.evidence_state.diagnostics.vector_health import compute_vector_health

    state = EvidenceState()
    state.features = [
        EvidenceFeature("query_complexity", 1.0, "retrieval", normalized_value=1.0),
        EvidenceFeature("recommended_topk", 10.0, "retrieval", normalized_value=1.0),
    ]
    compute_vector_health(state)
    assert state.health.health_score == 0.0


def test_health_is_capped_by_evidence_strength():
    from backend.evidence_state.state.evidence_state import EvidenceState, EvidenceFeature
    from backend.evidence_state.diagnostics.vector_health import compute_vector_health

    state = EvidenceState()
    state.features = [
        EvidenceFeature("agreement", 0.1, "retrieval", normalized_value=0.1),
        EvidenceFeature("graph_score", 0.1, "graph", normalized_value=0.1),
        EvidenceFeature("evidence_quality", 0.1, "quality", normalized_value=0.1),
    ]
    compute_vector_health(state)
    assert state.health.health_score <= 0.101


def test_zero_evidence_vector_is_maximally_uncertain():
    from backend.evidence_state.state.evidence_state import EvidenceState
    from backend.evidence_state.diagnostics.uncertainty import compute_uncertainty

    state = EvidenceState()
    state.weighted_vector = np.zeros(5, dtype=float)
    compute_uncertainty(state)
    assert state.uncertainty.uncertainty_score == 1.0
    assert state.signals.uncertainty.score == 1.0


def test_uncertainty_responds_to_evidence_strength():
    from backend.evidence_state.state.evidence_state import EvidenceState
    from backend.evidence_state.diagnostics.uncertainty import compute_uncertainty

    weak = EvidenceState()
    weak.weighted_vector = np.full(5, 0.1, dtype=float)
    compute_uncertainty(weak)

    strong = EvidenceState()
    strong.weighted_vector = np.full(5, 0.9, dtype=float)
    compute_uncertainty(strong)

    assert weak.uncertainty.uncertainty_score > strong.uncertainty.uncertainty_score


def test_adaptive_policy_does_not_call_confidence_drop_convergence():
    from backend.ms_arc.retrieval.adaptive_policy import AdaptiveRetrievalPolicy

    policy = AdaptiveRetrievalPolicy()
    assert not policy.should_stop(
        confidence=0.40,
        previous_confidence=0.70,
        current_k=5,
    )
    assert policy.should_stop(
        confidence=0.705,
        previous_confidence=0.70,
        current_k=5,
    )


def test_query_embedding_cache_reuses_embeddings_across_adaptive_depths(monkeypatch):
    module = _load_hybrid_search_with_stubs(monkeypatch)
    calls = []

    def counted_encode(query):
        calls.append(query)
        return np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)

    monkeypatch.setattr(module, "encode_query", counted_encode)
    module.clear_query_cache()
    first, first_status = module._get_query_embedding("same query")
    second, second_status = module._get_query_embedding("same query")
    assert len(calls) == 1
    assert first_status == "miss"
    assert second_status == "hit"
    assert np.array_equal(first, second)
    # Cached arrays are isolated from caller mutation.
    first[0, 0] = 999.0
    third, third_status = module._get_query_embedding("same query")
    assert third[0, 0] == 1.0
    assert third_status == "hit"
    info = module.query_cache_info()
    assert info["embedding_cache_hits"] == 2
    assert info["embedding_cache_misses"] == 1


def test_query_embedding_cache_is_bounded_and_namespaced(monkeypatch):
    module = _load_hybrid_search_with_stubs(monkeypatch)
    monkeypatch.setattr(module, "_QUERY_EMBEDDING_CACHE_MAX_SIZE", 2)
    module.clear_query_cache()
    for query in ("q1", "q2", "q3"):
        module._get_query_embedding(query)
    info = module.query_cache_info()
    assert info["embedding_size"] <= 2
    assert info["embedding_cache_evictions"] == 1
    key_a = module._embedding_model_cache_identity()
    # Explicit model identity/config namespace is part of the query cache key.
    assert isinstance(key_a, tuple) and "normalize=true" in key_a
    monkeypatch.setenv("RAGAR_EMBEDDING_MODEL_REVISION", "revision-test-2")
    key_b = module._embedding_model_cache_identity()
    assert key_a != key_b
