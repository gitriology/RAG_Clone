import json
from dataclasses import asdict

from backend.evidence_graph.serialization.graph_snapshot import (
    GraphSnapshot,
    SerializedNode,
    SerializedEdge,
    SerializedGraphStatistics,
    SerializedGraphSignals,
)


# ==========================================================
# BUILD SNAPSHOT
# ==========================================================

def build_snapshot(
    graph_state,
) -> GraphSnapshot:
    """
    Converts an EvidenceGraphState into a serializable snapshot.
    """

    snapshot = GraphSnapshot()

    # ======================================================
    # Nodes
    # ======================================================

    for node in graph_state.nodes:

        snapshot.nodes.append(

            SerializedNode(

                node_id=node.node_id,

                text=node.text,

                domain=node.domain,

                source=node.source,

                dense_score=node.dense_score,

                sparse_score=node.sparse_score,

                rerank_score=node.rerank_score,

                hybrid_score=node.hybrid_score,

                degree=node.degree_centrality,

                betweenness=node.betweenness_centrality,

                closeness=node.closeness_centrality,

                eigenvector=node.eigenvector_centrality,

                pagerank=node.pagerank,

                importance=node.importance_score,

            )

        )

    # ======================================================
    # Edges
    # ======================================================

    for edge in graph_state.edges:

        snapshot.edges.append(

            SerializedEdge(

                source=edge.source,

                target=edge.target,

                weight=edge.weight,

                relation=edge.relation,

            )

        )

    # ======================================================
    # Statistics
    # ======================================================

    snapshot.statistics = SerializedGraphStatistics(

        total_nodes=graph_state.statistics.total_nodes,

        total_edges=graph_state.statistics.total_edges,

        connected_components=graph_state.statistics.connected_components,

        average_degree=graph_state.statistics.average_degree,

        average_edge_weight=graph_state.statistics.average_edge_weight,

        graph_density=graph_state.statistics.graph_density,

    )

    # ======================================================
    # Signals
    # ======================================================

    snapshot.signals = SerializedGraphSignals(

        graph_score=graph_state.graph_score,

        coherence_score=graph_state.signals.coherence.score,

        evidence_quality=graph_state.signals.quality.score,

        average_degree=graph_state.signals.centrality.average_degree,

        average_betweenness=graph_state.signals.centrality.average_betweenness,

        average_closeness=graph_state.signals.centrality.average_closeness,

        average_eigenvector=graph_state.signals.centrality.average_eigenvector,

        average_pagerank=graph_state.signals.centrality.average_pagerank,

        graph_consensus=graph_state.signals.ranking.graph_consensus,

    )

    return snapshot


# ==========================================================
# SAVE GRAPH
# ==========================================================

def save_graph(
    graph_state,
    filename,
):
    """
    Saves the graph snapshot as JSON.
    """

    snapshot = build_snapshot(
        graph_state
    )

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(

            asdict(snapshot),

            file,

            indent=4,

            ensure_ascii=False,

        )