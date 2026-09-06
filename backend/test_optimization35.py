import networkx as nx
import pytest
from unittest.mock import patch

from backend.evidence_graph.analytics.graph_analysis import (
    ANALYTIC_METRICS,
    analyze_graph,
)
from backend.evidence_graph.analytics.node_importance import (
    compute_node_importance,
)
from backend.evidence_graph.models.evidence_node import EvidenceNode
from backend.evidence_graph.state.evidence_graph_state import EvidenceGraphState
from backend.ms_arc.state.retrieval_state import RetrievalState
from backend.evidence_graph.graph_builder import ensure_graph_analytics


def make_state(n=4):
    state = EvidenceGraphState()
    for i in range(n):
        node = EvidenceNode(
            node_id=str(i),
            text=f"evidence {i}",
            domain="test",
            source="test.pdf",
            rerank_score=0.5,
        )
        state.nodes.append(node)
    graph = nx.path_graph([str(i) for i in range(n)])
    for node in state.nodes:
        graph.nodes[node.node_id]["node"] = node
    state.graph = graph
    return state


def test_metric_selective_analysis_only_computes_requested_metrics():
    state = make_state()

    with patch(
        "backend.evidence_graph.analytics.graph_analysis.compute_degree",
        wraps=lambda g: nx.degree_centrality(g),
    ) as degree, patch(
        "backend.evidence_graph.analytics.graph_analysis.compute_pagerank",
        wraps=lambda g: nx.pagerank(g, alpha=0.85),
    ) as pagerank, patch(
        "backend.evidence_graph.analytics.graph_analysis.compute_betweenness",
    ) as betweenness, patch(
        "backend.evidence_graph.analytics.graph_analysis.compute_closeness",
    ) as closeness, patch(
        "backend.evidence_graph.analytics.graph_analysis.compute_eigenvector",
    ) as eigenvector:
        analyze_graph(state, metrics={"degree", "pagerank"})

    assert degree.call_count == 1
    assert pagerank.call_count == 1
    assert betweenness.call_count == 0
    assert closeness.call_count == 0
    assert eigenvector.call_count == 0
    assert state.analytics_metrics_computed == set()
    assert state.nodes[0].degree_centrality >= 0.0
    assert state.nodes[0].pagerank > 0.0


def test_ensure_graph_analytics_reuses_existing_metrics():
    state = make_state()
    retrieval = RetrievalState(query="test")

    with patch(
        "backend.evidence_graph.graph_builder.analyze_graph",
        wraps=__import__(
            "backend.evidence_graph.analytics.graph_analysis",
            fromlist=["analyze_graph"],
        ).analyze_graph,
    ) as analyze:
        ensure_graph_analytics(
            retrieval,
            state,
            metrics={"degree", "pagerank"},
        )
        ensure_graph_analytics(
            retrieval,
            state,
            metrics={"degree", "pagerank"},
        )

    assert analyze.call_count == 1
    assert state.analytics_metrics_computed == {"degree", "pagerank"}
    assert state.analytics_computed is False


def test_full_analytics_contract_remains_available():
    state = make_state()
    retrieval = RetrievalState(query="test")
    retrieval.signals.decision.confidence = 0.5

    result = ensure_graph_analytics(retrieval, state)

    assert result.analytics_computed is True
    assert result.analytics_metrics_computed == set(ANALYTIC_METRICS)
    assert result.ranking is not None
    assert result.graph_score >= 0.0


def test_node_importance_requires_explicit_metrics_and_is_derived_on_demand():
    state = make_state()

    with pytest.raises(RuntimeError, match="requires graph analytics metrics"):
        compute_node_importance(state)

    state.analytics_metrics_computed.update(ANALYTIC_METRICS)
    # Populate the historical centrality fields without invoking expensive
    # analytics in this unit test.
    for node in state.nodes:
        node.degree_centrality = 0.5
        node.betweenness_centrality = 0.1
        node.closeness_centrality = 0.2
        node.eigenvector_centrality = 0.3
        node.pagerank = 0.25

    compute_node_importance(state)
    assert state.signals.importance.average_importance > 0.0
    assert state.signals.importance.most_important_node != ""
