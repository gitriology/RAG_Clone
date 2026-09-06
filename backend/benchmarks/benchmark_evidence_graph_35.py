import time
import networkx as nx

from backend.evidence_graph.analytics.graph_analysis import analyze_graph
from backend.evidence_graph.models.evidence_node import EvidenceNode
from backend.evidence_graph.state.evidence_graph_state import EvidenceGraphState


def make_state(n=100):
    state = EvidenceGraphState()
    for i in range(n):
        state.nodes.append(
            EvidenceNode(
                node_id=str(i),
                text=f"evidence {i}",
                domain="benchmark",
                source="synthetic",
                rerank_score=0.5,
            )
        )
    state.graph = nx.path_graph([str(i) for i in range(n)])
    return state


def timed(metrics):
    state = make_state()
    start = time.perf_counter()
    analyze_graph(state, metrics=metrics)
    elapsed = time.perf_counter() - start
    return elapsed


def main():
    full = {"degree", "betweenness", "closeness", "eigenvector", "pagerank"}
    lightweight = {"degree", "pagerank"}

    full_time = timed(full)
    lazy_time = timed(lightweight)
    reduction = (1.0 - lazy_time / full_time) * 100.0 if full_time else 0.0

    print("Optimization #35 — Evidence Graph metric-level lazy analytics benchmark")
    print()
    print("Nodes: 100")
    print("Graph: path")
    print("Full metrics: degree, betweenness, closeness, eigenvector, pagerank")
    print("Lazy metrics: degree, pagerank")
    print(f"Full analytics time: {full_time:.6f}s")
    print(f"Lazy metric subset time: {lazy_time:.6f}s")
    print(f"Measured reduction: {reduction:.2f}%")
    print()
    print("Note: synthetic micro-benchmark; isolated graph analytics only, not end-to-end RAG latency.")


if __name__ == "__main__":
    main()
