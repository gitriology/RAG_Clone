from types import SimpleNamespace
from unittest.mock import patch

from backend.evidence_graph.graph_builder import build_graph, ensure_graph_analytics
from backend.evidence_graph.models.evidence_edge import EvidenceEdge
from backend.ms_arc.state.retrieval_state import RetrievalState


def make_retrieval(confidence=0.95, count=3):
    results = []
    for i in range(count):
        results.append(
            SimpleNamespace(
                doc_id=i,
                text=f"Evidence {i}",
                metadata={"domain": "test", "source": "test", "hybrid_score": 0.9},
                dense_score=0.9,
                sparse_score=0.9,
                rerank_score=0.9,
            )
        )
    state = RetrievalState(query="test")
    state.reranked_results = results
    state.signals.decision.confidence = confidence
    return state


def fake_edges(nodes, **kwargs):
    return [
        EvidenceEdge(source=0, target=1, weight=0.9, relation="semantic_similarity"),
        EvidenceEdge(source=1, target=2, weight=0.9, relation="semantic_similarity"),
        EvidenceEdge(source=0, target=2, weight=0.9, relation="semantic_similarity"),
    ]


def test_lazy_mode_skips_expensive_analytics_when_graph_is_sufficient():
    state = make_retrieval(confidence=0.95)

    with patch("backend.evidence_graph.graph_builder.build_edges", side_effect=fake_edges), \
         patch("backend.evidence_graph.graph_builder.analyze_graph") as analyze, \
         patch("backend.evidence_graph.graph_builder.build_graph_ranking") as ranking:
        graph = build_graph(state, analytics_mode="lazy")

    assert graph.analytics_mode == "lazy"
    assert graph.analytics_computed is False
    assert graph.analytics_sufficiency is True
    assert graph.analytics_skip_reason == "high_confidence_connected_graph"
    analyze.assert_not_called()
    ranking.assert_not_called()
    assert graph.graph.number_of_nodes() == 3
    assert graph.graph.number_of_edges() == 3


def test_lazy_mode_falls_back_to_full_analytics_when_evidence_is_ambiguous():
    state = make_retrieval(confidence=0.70)

    with patch("backend.evidence_graph.graph_builder.build_edges", side_effect=fake_edges), \
         patch("backend.evidence_graph.graph_builder.analyze_graph", wraps=__import__(
             "backend.evidence_graph.analytics.graph_analysis", fromlist=["analyze_graph"]
         ).analyze_graph) as analyze:
        graph = build_graph(state, analytics_mode="lazy")

    assert graph.analytics_mode == "lazy"
    assert graph.analytics_computed is True
    assert graph.analytics_sufficiency is False
    assert graph.analytics_skip_reason == "retrieval_confidence_below_lazy_threshold"
    analyze.assert_called_once()
    assert graph.ranking is not None


def test_full_mode_preserves_complete_analytics_contract():
    state = make_retrieval(confidence=0.95)

    with patch("backend.evidence_graph.graph_builder.build_edges", side_effect=fake_edges):
        graph = build_graph(state, analytics_mode="full")

    assert graph.analytics_mode == "full"
    assert graph.analytics_computed is True
    assert graph.analytics_skip_reason == "full_mode_requested"
    assert graph.ranking is not None
    assert 0.0 <= graph.graph_score <= 1.0


def test_ensure_graph_analytics_materializes_lazy_state_once():
    state = make_retrieval(confidence=0.95)

    with patch("backend.evidence_graph.graph_builder.build_edges", side_effect=fake_edges):
        graph = build_graph(state, analytics_mode="lazy")

    assert graph.analytics_computed is False

    with patch("backend.evidence_graph.graph_builder.analyze_graph", wraps=__import__(
        "backend.evidence_graph.analytics.graph_analysis", fromlist=["analyze_graph"]
    ).analyze_graph) as analyze:
        hydrated = ensure_graph_analytics(state, graph)
        again = ensure_graph_analytics(state, hydrated)

    assert hydrated is again
    assert hydrated.analytics_computed is True
    analyze.assert_called_once()
