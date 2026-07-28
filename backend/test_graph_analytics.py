from backend.ms_arc.state.retrieval_state import RetrievalState
from backend.ms_arc.retrieval.retrieve import retrieve
from backend.ms_arc.confidence.margin import compute_margin

from backend.evidence_graph.graph_builder import build_graph

state = RetrievalState(
    query="What is vaccination?"
)

state = retrieve(state)

state = compute_margin(state)

graph = build_graph(state)

print("="*60)
print("GRAPH ANALYTICS")
print("="*60)

for node in graph.nodes:

    print()

    print(node.node_id)

    print("Degree      :", round(node.degree_centrality,3))

    print("Betweenness :", round(node.betweenness_centrality,3))

    print("Closeness   :", round(node.closeness_centrality,3))

    print("Eigenvector :", round(node.eigenvector_centrality,3))

    print("PageRank    :", round(node.pagerank,3))

    print("Importance  :", round(node.importance_score,3))


print()

print("="*60)

print("Centrality")

print("="*60)

print(graph.signals.centrality)

print()

print("Importance")

print(graph.signals.importance)