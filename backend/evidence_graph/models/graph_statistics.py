from dataclasses import dataclass

import networkx as nx


@dataclass
class GraphStatistics:

    total_nodes: int = 0

    total_edges: int = 0

    connected_components: int = 0

    average_degree: float = 0.0

    average_edge_weight: float = 0.0

    graph_density: float = 0.0


def compute_graph_statistics(
    graph: nx.Graph,
) -> GraphStatistics:

    stats = GraphStatistics()

    stats.total_nodes = graph.number_of_nodes()

    stats.total_edges = graph.number_of_edges()

    stats.connected_components = nx.number_connected_components(
        graph
    )

    stats.graph_density = nx.density(
        graph
    )

    if stats.total_nodes:

        stats.average_degree = (

            2 * stats.total_edges

        ) / stats.total_nodes

    if stats.total_edges:

        weights = [

            data.get("weight", 0.0)

            for _, _, data in graph.edges(data=True)

        ]

        stats.average_edge_weight = (

            sum(weights) / len(weights)

        )

    return stats