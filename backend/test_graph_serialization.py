from backend.ms_arc.state.retrieval_state import RetrievalState
from backend.ms_arc.retrieval.retrieve import retrieve
from backend.ms_arc.confidence.margin import compute_margin

from backend.evidence_graph.graph_builder import build_graph

from backend.evidence_graph.serialization.graph_serializer import (
    save_graph,
)

from backend.evidence_graph.serialization.graph_deserializer import (
    load_snapshot,
)

state = RetrievalState(
    query="What is vaccination?"
)

state = retrieve(state)

state = compute_margin(state)

graph = build_graph(state)

save_graph(
    graph,
    "graph_snapshot.json",
)

snapshot = load_snapshot(
    "graph_snapshot.json",
)

print("=" * 60)
print("GRAPH SNAPSHOT")
print("=" * 60)

print("Nodes :", len(snapshot.nodes))
print("Edges :", len(snapshot.edges))
print("Score :", snapshot.signals.graph_score)