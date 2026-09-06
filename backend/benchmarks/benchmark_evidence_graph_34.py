"""Micro-benchmark for Optimization #34 lazy Evidence Graph analytics."""

from types import SimpleNamespace
from unittest.mock import patch
import time

from backend.evidence_graph.graph_builder import build_graph
from backend.evidence_graph.models.evidence_edge import EvidenceEdge
from backend.ms_arc.state.retrieval_state import RetrievalState


def make_state(confidence=0.95, count=100):
    state = RetrievalState(query="benchmark")
    state.signals.decision.confidence = confidence
    state.reranked_results = [
        SimpleNamespace(
            doc_id=i,
            text=f"Evidence {i}",
            metadata={"domain": "test", "source": "test", "hybrid_score": 0.9},
            dense_score=0.9,
            sparse_score=0.9,
            rerank_score=0.9,
        )
        for i in range(count)
    ]
    return state


def fake_edges(nodes, **kwargs):
    # Sparse chain keeps the benchmark deterministic and cheap.
    return [
        EvidenceEdge(
            source=i,
            target=i + 1,
            weight=0.9,
            relation="semantic_similarity",
        )
        for i in range(len(nodes) - 1)
    ]


def slow_analytics(graph_state):
    # Simulate downstream analytics work while preserving the state contract.
    time.sleep(0.002)
    for node in graph_state.nodes:
        node.degree_centrality = 0.1
        node.betweenness_centrality = 0.1
        node.closeness_centrality = 0.1
        node.eigenvector_centrality = 0.1
        node.pagerank = 1.0 / max(len(graph_state.nodes), 1)
        node.importance_score = 0.1
    graph_state.signals.importance.average_importance = 0.1
    graph_state.signals.importance.highest_importance_score = 0.1
    graph_state.signals.importance.most_important_node = graph_state.nodes[0].node_id
    return graph_state


def run(mode):
    state = make_state()
    start = time.perf_counter()
    with patch("backend.evidence_graph.graph_builder.build_edges", side_effect=fake_edges), \
         patch("backend.evidence_graph.graph_builder.analyze_graph", side_effect=slow_analytics):
        graph = build_graph(state, analytics_mode=mode)
    return graph, time.perf_counter() - start


if __name__ == "__main__":
    full_graph, full_time = run("full")
    lazy_graph, lazy_time = run("lazy")

    print("Optimization #34 — Evidence Graph analytics benchmark")
    print(f"Nodes: {len(full_graph.nodes)}")
    print(f"Edges: {len(full_graph.edges)}")
    print(f"Full analytics computed: {full_graph.analytics_computed}")
    print(f"Lazy analytics computed: {lazy_graph.analytics_computed}")
    print(f"Lazy skip reason: {lazy_graph.analytics_skip_reason}")
    print(f"Full mode time: {full_time:.6f}s")
    print(f"Lazy mode time: {lazy_time:.6f}s")
    if full_time > 0:
        print(f"Measured reduction: {(1 - lazy_time / full_time) * 100:.2f}%")
    print("Note: synthetic micro-benchmark; not end-to-end RAG latency.")
