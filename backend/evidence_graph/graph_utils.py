import networkx as nx


# ==========================================================
# GRAPH DENSITY
# ==========================================================

def compute_density(graph: nx.Graph) -> float:
    """
    Computes graph density.

    Density = Existing Edges / Possible Edges
    """

    if graph.number_of_nodes() <= 1:
        return 0.0

    return nx.density(graph)


# ==========================================================
# CONNECTED COMPONENTS
# ==========================================================

def connected_components(graph: nx.Graph):
    """
    Returns all connected components.
    """

    return list(nx.connected_components(graph))


# ==========================================================
# NUMBER OF COMPONENTS
# ==========================================================

def component_count(graph: nx.Graph) -> int:

    return nx.number_connected_components(graph)


# ==========================================================
# AVERAGE DEGREE
# ==========================================================

def average_degree(graph: nx.Graph) -> float:
    """
    Average node degree.
    """

    n = graph.number_of_nodes()

    if n == 0:
        return 0.0

    degrees = [degree for _, degree in graph.degree()]

    return sum(degrees) / n


# ==========================================================
# AVERAGE EDGE WEIGHT
# ==========================================================

def average_edge_weight(graph: nx.Graph) -> float:
    """
    Average similarity weight.
    """

    if graph.number_of_edges() == 0:
        return 0.0

    weights = []

    for _, _, data in graph.edges(data=True):

        weights.append(
            data.get("weight", 0.0)
        )

    return sum(weights) / len(weights)


# ==========================================================
# IS CONNECTED
# ==========================================================

def is_connected(graph: nx.Graph) -> bool:

    if graph.number_of_nodes() == 0:
        return False

    return nx.is_connected(graph)


# ==========================================================
# NODE CENTRALITY
# ==========================================================

def degree_centrality(graph: nx.Graph):
    """
    Degree centrality of every node.
    """

    return nx.degree_centrality(graph)


# ==========================================================
# BETWEENNESS CENTRALITY
# ==========================================================

def betweenness_centrality(graph: nx.Graph):
    """
    Betweenness centrality.
    """

    return nx.betweenness_centrality(
        graph,
        weight="weight",
    )


# ==========================================================
# CLOSENESS CENTRALITY
# ==========================================================

def closeness_centrality(graph: nx.Graph):
    """
    Closeness centrality.
    """

    return nx.closeness_centrality(graph)


# ==========================================================
# GRAPH SUMMARY
# ==========================================================

def summarize_graph(graph: nx.Graph) -> dict:
    """
    Returns all basic graph statistics.

    Used by graph_builder.py and graph_score.py.
    """

    return {

        "nodes":
            graph.number_of_nodes(),

        "edges":
            graph.number_of_edges(),

        "density":
            compute_density(graph),

        "components":
            component_count(graph),

        "average_degree":
            average_degree(graph),

        "average_edge_weight":
            average_edge_weight(graph),

        "connected":
            is_connected(graph),

    }