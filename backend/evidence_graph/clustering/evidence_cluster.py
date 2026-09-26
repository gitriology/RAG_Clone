import networkx as nx
from typing import List
from backend.evidence_graph.state.evidence_graph_state import (
    EvidenceGraphState,
)
from backend.evidence_graph.models.evidence_cluster import (
    EvidenceCluster,
)

# ==========================================================
# Cluster Score
# ==========================================================

def compute_cluster_score(
    graph: nx.Graph,
    nodes: List[str],
) -> float:
    """
    Cluster quality.

    Current score =
        average reranker confidence

    Future:
        combine
            • reranker
            • evidence agreement
            • graph centrality
            • semantic coherence
    """

    if not nodes:
        return 0.0

    total = 0.0

    for node in nodes:
        total += graph.nodes[node]["node"].rerank_score
    return total / len(nodes)


# ==========================================================
# Centroid
# ==========================================================

def find_centroid(
    graph: nx.Graph,
    nodes: List[str],
) -> str:
    """
    Picks the highest confidence document
    inside the cluster.

    Future versions may replace this with

    • PageRank
    • Betweenness Centrality
    • Eigenvector Centrality
    """

    if not nodes:
        return ""

    best_node = max(

        nodes,

        key=lambda n:

        graph.nodes[n]["node"].rerank_score,

    )

    return best_node


# ==========================================================
# Evidence Clustering
# ==========================================================

def build_clusters(
    graph_state: EvidenceGraphState,
) -> EvidenceGraphState:
    """
    Converts connected components into
    typed EvidenceCluster objects.
    """

    graph = graph_state.graph

    graph_state.clusters.clear()

    components = list(

        nx.connected_components(graph)

    )

    for cluster_id, component in enumerate(
        components
    ):

        nodes = sorted(component)

        centroid = find_centroid(
            graph,
            nodes,
        )

        score = compute_cluster_score(
            graph,
            nodes,
        )

        cluster = EvidenceCluster(

            cluster_id=cluster_id,

            node_ids=nodes,

            centroid_node=centroid,

            score=score,
        )

        graph_state.clusters.append(
            cluster
        )

    return graph_state


# ==========================================================
# Largest Cluster
# ==========================================================

def largest_cluster(
    graph_state: EvidenceGraphState,
):
    """
    Returns the largest evidence cluster.
    """

    if not graph_state.clusters:
        return None

    return max(

        graph_state.clusters,

        key=lambda c: len(c.node_ids),

    )


# ==========================================================
# Cluster Lookup
# ==========================================================

def cluster_of_document(
    graph_state: EvidenceGraphState,
    document_id: str,
):
    """
    Returns the cluster containing
    the given document.
    """

    for cluster in graph_state.clusters:

        if document_id in cluster.document_ids:

            return cluster

    return None


# ==========================================================
# Cluster Statistics
# ==========================================================

def average_cluster_size(
    graph_state: EvidenceGraphState,
) -> float:
    """
    Average number of documents
    inside each cluster.
    """

    if not graph_state.clusters:
        return 0.0

    total = sum(

        len(cluster.node_ids)

        for cluster in graph_state.clusters

    )

    return total / len(graph_state.clusters)


def cluster_count(
    graph_state: EvidenceGraphState,
) -> int:

    return len(graph_state.clusters)