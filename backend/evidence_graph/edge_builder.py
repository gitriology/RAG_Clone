from typing import List, Optional

import numpy as np

from backend.evidence_graph.models.evidence_node import EvidenceNode
from backend.evidence_graph.models.evidence_edge import EvidenceEdge
from backend.retrieval.dense.embedder import get_embeddings


# ==========================================================
# CONFIGURATION
# ==========================================================

DEFAULT_SIMILARITY_THRESHOLD = 0.55
DEFAULT_MAX_NEIGHBORS_PER_NODE = 3


# ==========================================================
# EMBEDDING VALIDATION
# ==========================================================

def _validate_embeddings(
    embeddings: np.ndarray,
    node_count: int,
) -> np.ndarray:
    """Validate and normalize the shape/dtype contract for node embeddings."""

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(
            "Evidence node embedding returned an unexpected "
            f"shape: {embeddings.shape}"
        )

    if embeddings.shape[0] != node_count:
        raise ValueError(
            "Number of node embeddings does not match "
            "number of nodes. "
            f"Nodes={node_count}, "
            f"Embeddings={embeddings.shape[0]}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Evidence node embeddings contain non-finite values.")

    # The normal production path already supplies normalized BGE embeddings.
    # Normalize defensively so precomputed embeddings have the same contract.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    zero_norm = norms.squeeze(1) <= 0.0
    if np.any(zero_norm):
        raise ValueError("Evidence node embeddings contain a zero vector.")

    return embeddings / norms


# ==========================================================
# SIMILARITY MATRIX
# ==========================================================

def compute_similarity_matrix(
    nodes: List[EvidenceNode],
    embeddings: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute the semantic similarity matrix for EvidenceGraph nodes.

    Optimization #20
    -----------------
    Encode all node texts in one shared BGE batch and compute cosine
    similarity with matrix multiplication.

    Optimization #33
    -----------------
    The embedding matrix may now be supplied by the caller. This lets a
    larger pipeline reuse embeddings that were already computed instead of
    invoking the embedding model again.
    """

    if not nodes:
        return np.empty((0, 0), dtype=np.float32)

    texts = []
    for node in nodes:
        if not isinstance(node.text, str):
            raise TypeError(
                "EvidenceNode.text must be a string. "
                f"Node {node.node_id} has type {type(node.text).__name__}."
            )
        texts.append(node.text)

    if embeddings is None:
        embeddings = get_embeddings(texts)

    embeddings = _validate_embeddings(embeddings, len(nodes))

    similarity_matrix = np.matmul(embeddings, embeddings.T)
    similarity_matrix = np.asarray(similarity_matrix, dtype=np.float32)
    return np.clip(similarity_matrix, -1.0, 1.0)


# ==========================================================
# SINGLE NODE SIMILARITY
# ==========================================================

def compute_similarity(
    node1: EvidenceNode,
    node2: EvidenceNode,
) -> float:
    """Compute semantic similarity between two EvidenceNode objects."""

    if not isinstance(node1, EvidenceNode):
        raise TypeError("node1 must be an EvidenceNode.")
    if not isinstance(node2, EvidenceNode):
        raise TypeError("node2 must be an EvidenceNode.")

    similarity_matrix = compute_similarity_matrix([node1, node2])
    return float(similarity_matrix[0, 1])


# ==========================================================
# BUILD EDGES
# ==========================================================

def build_edges(
    nodes: List[EvidenceNode],
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    max_neighbors_per_node: Optional[int] = DEFAULT_MAX_NEIGHBORS_PER_NODE,
    embeddings: Optional[np.ndarray] = None,
) -> List[EvidenceEdge]:
    """
    Build semantic EvidenceGraph edges using vectorized cosine similarity.

    Optimization #33: bounded semantic neighborhood
    --------------------------------------------------
    A threshold-only graph can create O(N^2) edge objects when many evidence
    nodes are mutually similar. The similarity matrix is still computed once,
    but edge construction is bounded by a configurable number of strongest
    neighbors per node.

    For every node, only its strongest ``max_neighbors_per_node`` semantic
    neighbors are eligible, subject to ``similarity_threshold``. An undirected
    pair is emitted once, so the final graph contains at most approximately
    N * max_neighbors_per_node / 2 edges (before mutual-neighborhood effects).

    ``max_neighbors_per_node=None`` preserves the previous threshold-only
    behavior for compatibility and ablation experiments.

    ``embeddings`` may be supplied when an upstream stage already computed the
    same normalized evidence-node embeddings, avoiding duplicate model calls.
    """

    if not isinstance(nodes, list):
        nodes = list(nodes)

    if similarity_threshold < -1.0 or similarity_threshold > 1.0:
        raise ValueError("similarity_threshold must be between -1.0 and 1.0.")

    if max_neighbors_per_node is not None:
        if not isinstance(max_neighbors_per_node, int):
            raise TypeError("max_neighbors_per_node must be an int or None.")
        if max_neighbors_per_node < 1:
            raise ValueError("max_neighbors_per_node must be >= 1 or None.")

    if len(nodes) < 2:
        return []

    similarity_matrix = compute_similarity_matrix(nodes, embeddings=embeddings)
    node_count = len(nodes)

    # Exclude self-similarity. Keep thresholding before neighbor selection so
    # weak semantic relations never become edges merely because they are among
    # the nearest available nodes.
    candidate_mask = similarity_matrix >= similarity_threshold
    np.fill_diagonal(candidate_mask, False)

    # ======================================================
    # Optimization #33: TOP-K NEIGHBOR SPARSIFICATION
    # ======================================================
    if max_neighbors_per_node is not None and max_neighbors_per_node < node_count - 1:
        selected_mask = np.zeros_like(candidate_mask, dtype=bool)

        for i in range(node_count):
            candidates = np.flatnonzero(candidate_mask[i])
            if candidates.size <= max_neighbors_per_node:
                selected_mask[i, candidates] = True
                continue

            scores = similarity_matrix[i, candidates]

            # argpartition avoids a full sort when the candidate pool is large.
            top_positions = np.argpartition(
                -scores,
                max_neighbors_per_node - 1,
            )[:max_neighbors_per_node]

            selected_candidates = candidates[top_positions]

            # Deterministic ordering is useful for reproducible graph output.
            selected_candidates = selected_candidates[
                np.argsort(-similarity_matrix[i, selected_candidates], kind="stable")
            ]

            selected_mask[i, selected_candidates] = True
    else:
        selected_mask = candidate_mask

    # An undirected graph should not depend on which endpoint selected the
    # other. Union the bounded neighborhoods so a strong one-way neighbor is
    # retained. Each pair is still emitted exactly once below.
    selected_mask = np.logical_or(selected_mask, selected_mask.T)

    pair_indices = np.triu_indices(node_count, k=1)
    pair_mask = selected_mask[pair_indices]
    matching_positions = np.flatnonzero(pair_mask)

    edges: List[EvidenceEdge] = []

    for position in matching_positions:
        i = int(pair_indices[0][position])
        j = int(pair_indices[1][position])
        similarity = float(similarity_matrix[i, j])

        edges.append(
            EvidenceEdge(
                source=nodes[i].node_id,
                target=nodes[j].node_id,
                weight=similarity,
                relation="semantic_similarity",
            )
        )

    return edges
