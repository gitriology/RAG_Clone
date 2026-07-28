import networkx as nx

from backend.ms_arc.state.retrieval_state import RetrievalState

from backend.evidence_graph.state.evidence_graph_state import (
    EvidenceGraphState,
)
from backend.evidence_graph.models.evidence_node import (
    EvidenceNode,
)

from backend.evidence_graph.edge_builder import build_edges

from backend.evidence_graph.models.graph_statistics import (
    compute_graph_statistics,
)

from backend.evidence_graph.clustering.evidence_cluster import (
    build_clusters,
)

from backend.evidence_graph.analytics.graph_analysis import (
    analyze_graph,
)

from backend.evidence_graph.quality.graph_coherence import (
    compute_graph_coherence,
)

from backend.evidence_graph.quality.evidence_quality import (
    compute_evidence_quality,
)

from backend.evidence_graph.graph_score import (
    compute_graph_score,
)

from backend.evidence_graph.ranking.graph_ranking import (
    build_graph_ranking,
)
from backend.evidence_graph.validation.graph_validation import (
    validate_graph,
)


def build_graph(
    retrieval_state: RetrievalState,
) -> EvidenceGraphState:
    """
    Complete Evidence Graph construction pipeline.

    Pipeline

        1. Create Nodes
        2. Create Edges
        3. Compute Graph Statistics
        4. Discover Clusters
        5. Compute Graph Coherence
        6. Compute Graph Analytics
        7. Compute Evidence Quality
        8. Graph-based Evidence Ranking
        9. Compute Overall Graph Score
    """

    graph_state = EvidenceGraphState()

    graph = nx.Graph()

    # =====================================================
    # Create Nodes
    # =====================================================

    evidence_nodes = []

    for document in retrieval_state.reranked_results:

        node = EvidenceNode(

            node_id=document.doc_id,

            text=document.text,

            domain=document.metadata.get(
                "domain",
                "unknown",
            ),

            source=document.metadata.get(
                "source",
                "unknown",
            ),

            dense_score=document.dense_score,

            sparse_score=document.sparse_score,

            rerank_score=document.rerank_score,

            hybrid_score=document.metadata.get(
                "hybrid_score",
                0.0,
            ),

        )

        evidence_nodes.append(node)

        graph_state.nodes.append(node)

        graph.add_node(

            node.node_id,

            node=node,

        )

        graph_state.node_lookup[
            node.node_id
        ] = node

    # =====================================================
    # Build Edges
    # =====================================================

    edges = build_edges(evidence_nodes)

    for edge in edges:

        graph.add_edge(

            edge.source,

            edge.target,

            weight=edge.weight,

            relation=edge.relation,

        )

        graph_state.edges.append(edge)

        graph_state.edge_lookup[
            (edge.source, edge.target)
        ] = edge

    graph_state.graph = graph

    # =====================================================
    # Graph Statistics
    # =====================================================

    graph_state.statistics = compute_graph_statistics(
        graph
    )

    graph_state.signals.graph.node_count = (
        graph.number_of_nodes()
    )

    graph_state.signals.graph.edge_count = (
        graph.number_of_edges()
    )

    graph_state.signals.graph.graph_density = (
        graph_state.statistics.graph_density
    )

    graph_state.signals.graph.average_degree = (
        graph_state.statistics.average_degree
    )

    graph_state.signals.graph.connected_components = (
        graph_state.statistics.connected_components
    )

    # =====================================================
    # Clustering
    # =====================================================

    graph_state = build_clusters(
        graph_state
    )

    # =====================================================
    # Graph Coherence
    # =====================================================

    graph_state = compute_graph_coherence(
        graph_state
    )

    # =====================================================
    # Graph Analytics
    # =====================================================

    graph_state = analyze_graph(
        graph_state
    )

    # =====================================================
    # Evidence Quality
    # =====================================================

    graph_state = compute_evidence_quality(

        retrieval_state,

        graph_state,

    )

    # =====================================================
    # Graph-based Evidence Ranking (Phase 8E)
    # =====================================================

    graph_ranking = build_graph_ranking(

        graph_state,

    )

    # Store ranking metrics

    graph_state.signals.ranking.average_score = (

        graph_ranking.statistics.average_score

    )

    graph_state.signals.ranking.highest_score = (

        graph_ranking.statistics.highest_score

    )

    graph_state.signals.ranking.lowest_score = (

        graph_ranking.statistics.lowest_score

    )

    graph_state.signals.ranking.graph_consensus = (

        graph_ranking.consensus.score

    )

    # (Optional but recommended)
    # Keep the full ranking state for Phase 9

    graph_state.ranking = graph_ranking

    # =====================================================
    # Overall Graph Score
    # =====================================================

    graph_state.graph_score = compute_graph_score(
        graph_state
    )
    graph_state.validation = validate_graph(
        graph_state
    )

    return graph_state