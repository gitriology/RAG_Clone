import networkx as nx


def compute_degree(graph):

    return nx.degree_centrality(graph)


def compute_betweenness(graph):

    return nx.betweenness_centrality(
        graph,
        normalized=True,
    )


def compute_closeness(graph):

    return nx.closeness_centrality(graph)


def compute_pagerank(graph):

    if graph.number_of_nodes() == 0:

        return {}

    return nx.pagerank(
        graph,
        alpha=0.85,
    )