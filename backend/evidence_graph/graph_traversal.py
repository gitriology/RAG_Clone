from collections import deque
from typing import List, Set

import networkx as nx

from backend.evidence_graph.state.evidence_graph_state import (
    EvidenceGraphState,
)


# ==========================================================
# BFS
# ==========================================================

def bfs(
    graph_state: EvidenceGraphState,
    start_node: str,
) -> List[str]:
    """
    Breadth-first traversal.

    Returns visitation order.
    """

    graph = graph_state.graph

    if start_node not in graph:
        return []

    visited: Set[str] = set()

    queue = deque([start_node])

    order = []

    while queue:

        node = queue.popleft()

        if node in visited:
            continue

        visited.add(node)

        order.append(node)

        for neighbor in graph.neighbors(node):

            if neighbor not in visited:
                queue.append(neighbor)

    return order


# ==========================================================
# DFS
# ==========================================================

def dfs(
    graph_state: EvidenceGraphState,
    start_node: str,
) -> List[str]:
    """
    Depth-first traversal.

    Returns visitation order.
    """

    graph = graph_state.graph

    if start_node not in graph:
        return []

    visited = set()

    order = []

    stack = [start_node]

    while stack:

        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)

        order.append(node)

        neighbors = list(graph.neighbors(node))

        neighbors.reverse()

        stack.extend(neighbors)

    return order


# ==========================================================
# SHORTEST PATH
# ==========================================================

def shortest_path(
    graph_state: EvidenceGraphState,
    source: str,
    target: str,
) -> List[str]:
    """
    Returns shortest path between two evidence nodes.
    """

    graph = graph_state.graph

    try:

        return nx.shortest_path(
            graph,
            source,
            target,
        )

    except nx.NetworkXNoPath:

        return []

    except nx.NodeNotFound:

        return []


# ==========================================================
# SUPPORT CHAIN
# ==========================================================

def support_chain(
    graph_state: EvidenceGraphState,
    node_id: str,
) -> List[str]:
    """
    Returns all evidence connected to one node.

    Useful for explanation generation.
    """

    graph = graph_state.graph

    if node_id not in graph:
        return []

    component = nx.node_connected_component(
        graph,
        node_id,
    )

    return list(component)


# ==========================================================
# NEIGHBORS
# ==========================================================

def neighbors(
    graph_state: EvidenceGraphState,
    node_id: str,
) -> List[str]:
    """
    Returns neighboring evidence nodes.
    """

    graph = graph_state.graph

    if node_id not in graph:
        return []

    return list(graph.neighbors(node_id))


# ==========================================================
# K-HOP NEIGHBORHOOD
# ==========================================================

def k_hop_neighbors(
    graph_state: EvidenceGraphState,
    node_id: str,
    hops: int = 2,
) -> List[str]:
    """
    Returns every node reachable within K hops.
    """

    graph = graph_state.graph

    if node_id not in graph:
        return []

    visited = {node_id}

    frontier = {node_id}

    for _ in range(hops):

        next_frontier = set()

        for node in frontier:

            next_frontier.update(graph.neighbors(node))

        next_frontier -= visited

        visited.update(next_frontier)

        frontier = next_frontier

    visited.remove(node_id)

    return list(visited)


# ==========================================================
# IS CONNECTED
# ==========================================================

def is_connected(
    graph_state: EvidenceGraphState,
) -> bool:
    """
    Returns whether the graph is fully connected.
    """

    graph = graph_state.graph

    if graph.number_of_nodes() == 0:
        return False

    return nx.is_connected(graph)


# ==========================================================
# DEGREE RANKING
# ==========================================================

def top_degree_nodes(
    graph_state: EvidenceGraphState,
    top_k: int = 5,
):
    """
    Returns nodes ranked by degree.
    """

    graph = graph_state.graph

    ranking = sorted(
        graph.degree(),
        key=lambda x: x[1],
        reverse=True,
    )

    return ranking[:top_k]