from unittest.mock import patch

import numpy as np

from backend.evidence_graph.edge_builder import (
    build_edges,
    compute_similarity_matrix,
)
from backend.evidence_graph.models.evidence_node import EvidenceNode


def make_nodes(count=5):
    return [
        EvidenceNode(
            node_id=i,
            text=f"Evidence sentence {i}",
            domain="test",
            source="test",
        )
        for i in range(count)
    ]


def test_precomputed_embeddings_are_reused():
    nodes = make_nodes(3)
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    with patch(
        "backend.evidence_graph.edge_builder.get_embeddings"
    ) as mock_get_embeddings:
        matrix = compute_similarity_matrix(nodes, embeddings=embeddings)

    mock_get_embeddings.assert_not_called()
    assert matrix.shape == (3, 3)
    assert np.isclose(matrix[0, 1], 0.8, atol=1e-5)
    assert np.isclose(matrix[1, 2], 0.6, atol=1e-5)


def test_top_k_neighbor_sparsification_bounds_edges():
    nodes = make_nodes(6)

    # Every pair is above the threshold. Without the #33 neighborhood cap,
    # the complete graph would contain 15 edges. With k=2, each node selects
    # only its two strongest neighbors; the undirected union remains sparse.
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.98, 0.20, 0.0, 0.0, 0.0, 0.0],
            [0.95, 0.31, 0.0, 0.0, 0.0, 0.0],
            [0.90, 0.435, 0.0, 0.0, 0.0, 0.0],
            [0.85, 0.527, 0.0, 0.0, 0.0, 0.0],
            [0.80, 0.60, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    edges = build_edges(
        nodes,
        similarity_threshold=0.55,
        max_neighbors_per_node=2,
        embeddings=embeddings,
    )

    # Six nodes * two directed neighborhood selections, then unioned into an
    # undirected graph: no node may select more than two outgoing neighbors.
    selected_pairs = set()
    for edge in edges:
        selected_pairs.add(tuple(sorted((edge.source, edge.target))))

    assert len(edges) <= 9
    assert len(selected_pairs) == len(edges)
    assert all(edge.relation == "semantic_similarity" for edge in edges)
    assert all(edge.weight >= 0.55 for edge in edges)


def test_legacy_threshold_only_mode_remains_available():
    nodes = make_nodes(3)
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.4358899],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    edges = build_edges(
        nodes,
        similarity_threshold=0.55,
        max_neighbors_per_node=None,
        embeddings=embeddings,
    )

    assert any(
        {edge.source, edge.target} == {0, 1}
        for edge in edges
    )
    assert not any(
        {edge.source, edge.target} == {0, 2}
        for edge in edges
    )
    assert not any(
        {edge.source, edge.target} == {1, 2}
        for edge in edges
    )
