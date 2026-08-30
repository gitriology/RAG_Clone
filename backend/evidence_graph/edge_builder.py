from typing import List

import numpy as np

from backend.evidence_graph.models.evidence_node import (
    EvidenceNode,
)
from backend.evidence_graph.models.evidence_edge import (
    EvidenceEdge,
)
from backend.retrieval.dense.embedder import (
    get_embeddings,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

DEFAULT_SIMILARITY_THRESHOLD = 0.55


# ==========================================================
# SIMILARITY MATRIX
# ==========================================================

def compute_similarity_matrix(
    nodes: List[EvidenceNode],
) -> np.ndarray:
    """
    Compute the semantic similarity matrix for EvidenceGraph nodes.

    Optimization #20
    -----------------
    Instead of encoding every node pair independently, all node
    texts are encoded exactly once using the shared BGE embedding
    model.

    Pipeline:

        node texts
            ↓
        one batched embedding pass
            ↓
        normalized embedding matrix
            ↓
        matrix multiplication
            ↓
        cosine similarity matrix

    Because get_embeddings() returns normalized embeddings,
    cosine similarity is equivalent to the dot product.

    Returns
    -------
    np.ndarray
        Float32 similarity matrix with shape:

            (number_of_nodes, number_of_nodes)

    Notes
    -----
    Diagonal values represent self-similarity and are therefore
    approximately 1.0.

    The matrix is symmetric:

        similarity[i, j] == similarity[j, i]
    """

    if not nodes:
        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    texts = []

    for node in nodes:
        if not isinstance(node.text, str):
            raise TypeError(
                "EvidenceNode.text must be a string. "
                f"Node {node.node_id} has type "
                f"{type(node.text).__name__}."
            )

        texts.append(node.text)

    # ------------------------------------------------------
    # SINGLE BATCH EMBEDDING PASS
    # ------------------------------------------------------
    embeddings = get_embeddings(texts)

    # ------------------------------------------------------
    # SHAPE VALIDATION
    # ------------------------------------------------------
    if embeddings.ndim != 2:
        raise ValueError(
            "Evidence node embedding returned an unexpected "
            f"shape: {embeddings.shape}"
        )

    if embeddings.shape[0] != len(nodes):
        raise ValueError(
            "Number of node embeddings does not match "
            "number of nodes. "
            f"Nodes={len(nodes)}, "
            f"Embeddings={embeddings.shape[0]}"
        )

    # ------------------------------------------------------
    # COSINE SIMILARITY
    # ------------------------------------------------------
    #
    # get_embeddings() already returns normalized vectors.
    #
    # Therefore:
    #
    #     cosine(E_i, E_j)
    #
    # is simply:
    #
    #     E_i dot E_j
    #
    similarity_matrix = np.matmul(
        embeddings,
        embeddings.T,
    )

    similarity_matrix = np.asarray(
        similarity_matrix,
        dtype=np.float32,
    )

    # ------------------------------------------------------
    # NUMERICAL SAFETY
    # ------------------------------------------------------
    #
    # Floating-point arithmetic can very occasionally produce
    # values slightly outside [-1, 1].
    #
    # Cosine similarity is mathematically bounded by this range.
    #
    similarity_matrix = np.clip(
        similarity_matrix,
        -1.0,
        1.0,
    )

    return similarity_matrix


# ==========================================================
# SINGLE NODE SIMILARITY
# ==========================================================

def compute_similarity(
    node1: EvidenceNode,
    node2: EvidenceNode,
) -> float:
    """
    Compute semantic similarity between two EvidenceNode objects.

    This function is retained for compatibility with existing
    callers.

    Optimization #20
    -----------------
    The main build_edges() pipeline does NOT call this function
    repeatedly. It computes all node embeddings once and derives
    the complete similarity matrix using vectorized matrix
    multiplication.

    For an isolated pairwise call, this function performs one
    batched embedding operation for the two node texts.
    """

    if not isinstance(node1, EvidenceNode):
        raise TypeError(
            "node1 must be an EvidenceNode."
        )

    if not isinstance(node2, EvidenceNode):
        raise TypeError(
            "node2 must be an EvidenceNode."
        )

    similarity_matrix = compute_similarity_matrix(
        [node1, node2]
    )

    return float(
        similarity_matrix[0, 1]
    )


# ==========================================================
# BUILD EDGES
# ==========================================================

def build_edges(
    nodes: List[EvidenceNode],
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> List[EvidenceEdge]:
    """
    Build semantic EvidenceGraph edges using vectorized
    cosine similarity.

    Optimization #20
    -----------------
    Previous implementation:

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                ...

    The previous implementation performed Python-level
    pair iteration and used weak similarity signals:

        same domain
        +
        reranker score similarity

    This implementation instead performs:

        node texts
            ↓
        one batched BGE encoding
            ↓
        embedding matrix
            ↓
        cosine similarity matrix
            ↓
        threshold
            ↓
        EvidenceEdge objects

    This removes repeated semantic model inference and moves
    similarity computation into optimized NumPy matrix
    operations.

    Parameters
    ----------
    nodes:
        EvidenceGraph nodes.

    similarity_threshold:
        Minimum cosine similarity required to create an edge.

    Returns
    -------
    List[EvidenceEdge]
        Semantic similarity edges.
    """

    # ------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------

    if not isinstance(nodes, list):
        nodes = list(nodes)

    if similarity_threshold < -1.0:
        raise ValueError(
            "similarity_threshold cannot be less than -1.0."
        )

    if similarity_threshold > 1.0:
        raise ValueError(
            "similarity_threshold cannot be greater than 1.0."
        )

    # ------------------------------------------------------
    # EMPTY / SINGLE NODE GRAPH
    # ------------------------------------------------------

    if len(nodes) < 2:
        return []

    # ------------------------------------------------------
    # COMPUTE ALL SIMILARITIES IN ONE PASS
    # ------------------------------------------------------

    similarity_matrix = compute_similarity_matrix(
        nodes
    )

    # ------------------------------------------------------
    # EXTRACT ONLY THE UPPER TRIANGLE
    # ------------------------------------------------------
    #
    # The similarity matrix is symmetric:
    #
    #       S[i,j] == S[j,i]
    #
    # We only need each pair once.
    #
    # k=1 excludes the diagonal because a node should not
    # create an edge with itself.
    #
    pair_indices = np.triu_indices(
        len(nodes),
        k=1,
    )

    pair_similarities = similarity_matrix[
        pair_indices
    ]

    # ------------------------------------------------------
    # APPLY THRESHOLD USING VECTORIZED NUMPY OPERATIONS
    # ------------------------------------------------------

    matching_positions = np.flatnonzero(
        pair_similarities >= similarity_threshold
    )

    # ------------------------------------------------------
    # BUILD EDGES
    # ------------------------------------------------------

    edges: List[EvidenceEdge] = []

    for position in matching_positions:
        i = int(pair_indices[0][position])
        j = int(pair_indices[1][position])

        similarity = float(
            pair_similarities[position]
        )

        edges.append(
            EvidenceEdge(
                source=nodes[i].node_id,
                target=nodes[j].node_id,
                weight=similarity,
                relation="semantic_similarity",
            )
        )

    return edges