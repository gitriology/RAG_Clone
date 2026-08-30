from unittest.mock import patch

import numpy as np

from backend.evidence_graph.edge_builder import build_edges
from backend.evidence_graph.models.evidence_node import EvidenceNode


def create_test_nodes():
    return [
        EvidenceNode(
            node_id=1,
            text="Machine learning is a field of artificial intelligence.",
            domain="AI",
            source="test",
            dense_score=0.9,
            sparse_score=0.8,
            rerank_score=0.9,
            hybrid_score=0.85,
        ),
        EvidenceNode(
            node_id=2,
            text="Artificial intelligence includes machine learning.",
            domain="AI",
            source="test",
            dense_score=0.8,
            sparse_score=0.7,
            rerank_score=0.8,
            hybrid_score=0.75,
        ),
        EvidenceNode(
            node_id=3,
            text="Vaccination helps protect people from infectious diseases.",
            domain="Health",
            source="test",
            dense_score=0.7,
            sparse_score=0.6,
            rerank_score=0.7,
            hybrid_score=0.65,
        ),
    ]


def main():
    nodes = create_test_nodes()

    # ------------------------------------------------------
    # Fake normalized embeddings
    # ------------------------------------------------------
    fake_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.4358899, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # ------------------------------------------------------
    # Optimization #20 verification
    # ------------------------------------------------------
    #
    # Patch the function where edge_builder actually uses it.
    #
    with patch(
        "backend.evidence_graph.edge_builder.get_embeddings",
        return_value=fake_embeddings,
    ) as mock_get_embeddings:

        edges = build_edges(
            nodes,
            similarity_threshold=0.55,
        )

    # ------------------------------------------------------
    # Verify embedding call count
    # ------------------------------------------------------

    assert mock_get_embeddings.call_count == 1, (
        "Optimization #20 failed: embeddings should be "
        "computed exactly once for the complete node set."
    )

    # ------------------------------------------------------
    # Verify all node texts were passed in one call
    # ------------------------------------------------------

    call_args = mock_get_embeddings.call_args

    encoded_texts = call_args.args[0]

    assert len(encoded_texts) == len(nodes), (
        "All evidence node texts must be encoded in the "
        "single embedding call."
    )

    assert encoded_texts == [node.text for node in nodes], (
        "The embedding call must receive the node texts "
        "in the original node order."
    )

    # ------------------------------------------------------
    # Verify vectorized similarity produced an edge
    # ------------------------------------------------------

    # Similarity between node 1 and node 2 is approximately
    # 0.9, therefore it should pass the 0.55 threshold.
    assert any(
        edge.source == 1 and edge.target == 2
        for edge in edges
    ), (
        "Expected a semantic edge between nodes 1 and 2."
    )

    # Node 3 should not connect to nodes 1 or 2 because
    # its embedding is orthogonal to them.
    assert not any(
        {
            edge.source,
            edge.target,
        }
        == {1, 3}
        for edge in edges
    )

    assert not any(
        {
            edge.source,
            edge.target,
        }
        == {2, 3}
        for edge in edges
    )

    print("=" * 60)
    print("GRAPH EMBEDDING OPTIMIZATION #20")
    print("=" * 60)

    print(f"Nodes tested              : {len(nodes)}")
    print(f"Embedding calls           : {mock_get_embeddings.call_count}")
    print(f"Expected embedding calls  : 1")
    print(f"Edges produced            : {len(edges)}")

    print()
    print("Embedding batching        : PASS")
    print("Single model call         : PASS")
    print("All node texts encoded    : PASS")
    print("Vectorized similarity     : PASS")
    print("Threshold filtering       : PASS")
    print()
    print("Optimization #20          : PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
