from backend.ms_arc.state.retrieval_state import RetrievalState

from backend.ms_arc.retrieval.retrieve import retrieve

from backend.ms_arc.confidence.margin import compute_margin

from backend.evidence_graph.graph_builder import build_graph


state = RetrievalState(
    query="What is vaccination?"
)

state.recommended_topk = 5

state = retrieve(state)

state = compute_margin(state)

graph_state = build_graph(state)

print()

print("=" * 60)

print("EVIDENCE GRAPH")

print("=" * 60)

print()

print("Nodes :", graph_state.statistics.total_nodes)

print("Edges :", graph_state.statistics.total_edges)

print("Density :", round(graph_state.statistics.graph_density, 4))

print("Average Degree :", round(graph_state.statistics.average_degree, 4))

print()

print("Node Lookup Type :", type(
    graph_state.node_lookup[
        graph_state.nodes[0].node_id
    ]
).__name__)

print()

for edge in graph_state.edges:

    print(
        edge.source,
        "<->",
        edge.target,
        round(edge.weight, 3),
        edge.relation,
    )