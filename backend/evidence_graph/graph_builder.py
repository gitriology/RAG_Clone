from __future__ import annotations

import networkx as nx

from backend.ms_arc.state.retrieval_state import RetrievalState

from backend.evidence_graph.state.evidence_graph_state import EvidenceGraphState
from backend.evidence_graph.models.evidence_node import EvidenceNode
from backend.evidence_graph.edge_builder import build_edges
from backend.evidence_graph.models.graph_statistics import compute_graph_statistics
from backend.evidence_graph.clustering.evidence_cluster import build_clusters
from backend.evidence_graph.analytics.graph_analysis import (
    analyze_graph,
    ANALYTIC_METRICS,
)
from backend.evidence_graph.quality.graph_coherence import compute_graph_coherence
from backend.evidence_graph.quality.evidence_quality import compute_evidence_quality
from backend.evidence_graph.graph_score import compute_graph_score
from backend.evidence_graph.ranking.graph_ranking import build_graph_ranking
from backend.evidence_graph.validation.graph_validation import validate_graph


VALID_ANALYTICS_MODES = {"full", "lazy"}
DEFAULT_ANALYTICS_MODE = "full"


def _cheap_sufficiency_check(
    retrieval_state: RetrievalState,
    graph_state: EvidenceGraphState,
) -> tuple[bool, str]:
    """
    Cheap pre-check used by Optimization #34.

    The check deliberately uses only already-available retrieval and graph
    topology signals. It never invokes centrality, PageRank, clustering
    analytics, ranking, or other expensive downstream analysis.

    A conservative policy is used: only a high-confidence, fully connected
    graph with at least one semantic relationship can bypass the expensive
    analytics stage. Ambiguous cases fall through to the full pipeline.
    """

    retrieval_confidence = float(
        getattr(
            retrieval_state.signals.decision,
            "confidence",
            0.0,
        )
    )

    node_count = graph_state.graph.number_of_nodes()
    edge_count = graph_state.graph.number_of_edges()
    components = graph_state.statistics.connected_components

    if node_count <= 1:
        return True, "single_node_graph"

    if retrieval_confidence < 0.90:
        return False, "retrieval_confidence_below_lazy_threshold"

    if edge_count == 0:
        return False, "no_semantic_edges"

    if components != 1:
        return False, "graph_not_connected"

    return True, "high_confidence_connected_graph"


def _finish_full_analytics_after_metrics(
    retrieval_state: RetrievalState,
    graph_state: EvidenceGraphState,
) -> EvidenceGraphState:
    """Materialize downstream analytics after all centrality metrics exist."""

    graph_state = compute_evidence_quality(
        retrieval_state,
        graph_state,
    )

    graph_ranking = build_graph_ranking(graph_state)

    graph_state.signals.ranking.average_score = graph_ranking.statistics.average_score
    graph_state.signals.ranking.highest_score = graph_ranking.statistics.highest_score
    graph_state.signals.ranking.lowest_score = graph_ranking.statistics.lowest_score
    graph_state.signals.ranking.graph_consensus = graph_ranking.consensus.score
    graph_state.ranking = graph_ranking

    graph_state.graph_score = compute_graph_score(graph_state)
    graph_state.validation = validate_graph(graph_state)
    graph_state.analytics_computed = True
    return graph_state


def _compute_full_analytics(
    retrieval_state: RetrievalState,
    graph_state: EvidenceGraphState,
) -> EvidenceGraphState:
    """Complete the historical full analytics contract."""

    graph_state = analyze_graph(graph_state, metrics=ANALYTIC_METRICS)
    graph_state.analytics_metrics_computed.update(ANALYTIC_METRICS)
    return _finish_full_analytics_after_metrics(
        retrieval_state,
        graph_state,
    )


def ensure_graph_analytics(
    retrieval_state: RetrievalState,
    graph_state: EvidenceGraphState,
    metrics=None,
) -> EvidenceGraphState:
    """
    Lazily materialize only the requested graph analytics.

    Optimization #35 makes graph analytics metric-level lazy: callers can
    request a small subset such as ``{"degree", "pagerank"}`` without
    paying for betweenness, closeness, and eigenvector centrality.

    ``metrics=None`` requests the complete historical analytics contract.
    Repeated requests are idempotent and already-computed metrics are reused.
    """

    if retrieval_state is None:
        raise ValueError(
            "ensure_graph_analytics() received retrieval_state=None."
        )

    if graph_state is None:
        raise ValueError(
            "ensure_graph_analytics() received graph_state=None."
        )

    requested = set(ANALYTIC_METRICS if metrics is None else metrics)
    requested = {str(metric).lower() for metric in requested}
    unknown = requested - ANALYTIC_METRICS
    if unknown:
        raise ValueError(
            f"Unsupported graph analytic metrics: {sorted(unknown)}. "
            f"Expected subset of {sorted(ANALYTIC_METRICS)}."
        )

    missing = requested - set(graph_state.analytics_metrics_computed)
    if missing:
        graph_state = analyze_graph(
            graph_state,
            metrics=missing,
        )
        graph_state.analytics_metrics_computed.update(missing)

    if requested == ANALYTIC_METRICS and not graph_state.analytics_computed:
        # Once every metric exists, materialize the historical downstream
        # quality/ranking/score contract exactly once.
        graph_state = _finish_full_analytics_after_metrics(
            retrieval_state,
            graph_state,
        )

    return graph_state


def build_graph(
    retrieval_state: RetrievalState,
    *,
    analytics_mode: str = DEFAULT_ANALYTICS_MODE,
) -> EvidenceGraphState:
    """
    Build the Evidence Graph with optional lazy analytics.

    Optimization #34
    ----------------
    ``analytics_mode='lazy'`` builds the basic graph first, performs a cheap
    sufficiency check, and only materializes expensive graph analytics when
    the evidence is ambiguous. Downstream callers can explicitly call
    ``ensure_graph_analytics()`` when centrality/ranking/graph-score details
    are required.

    ``analytics_mode='full'`` preserves the historical complete graph-analysis
    contract and is the default for backward compatibility.

    Optimization #20
    ----------------
    Edge construction uses batched semantic embeddings and a vectorized
    similarity matrix.

    Optimization #21
    ----------------
    The existing MS-ARC RetrievalState is reused directly; no second
    reranking operation is performed here.
    """

    if retrieval_state is None:
        raise ValueError(
            "build_graph() received retrieval_state=None."
        )

    if analytics_mode not in VALID_ANALYTICS_MODES:
        raise ValueError(
            f"Unsupported analytics_mode={analytics_mode!r}. "
            f"Expected one of {sorted(VALID_ANALYTICS_MODES)}."
        )

    if not retrieval_state.reranked_results:
        raise ValueError(
            "build_graph() received a RetrievalState "
            "with no reranked_results."
        )

    graph_state = EvidenceGraphState()
    graph_state.analytics_mode = analytics_mode
    graph_state.analytics_computed = False
    graph_state.analytics_metrics_computed.clear()

    graph = nx.Graph()
    evidence_nodes = []

    # ======================================================
    # Create Nodes
    # ======================================================
    for document in retrieval_state.reranked_results:
        if document is None:
            continue

        metadata = document.metadata
        if not isinstance(metadata, dict):
            metadata = {}

        node = EvidenceNode(
            node_id=document.doc_id,
            text=document.text,
            domain=metadata.get("domain", "unknown"),
            source=metadata.get("source", "unknown"),
            dense_score=document.dense_score,
            sparse_score=document.sparse_score,
            rerank_score=document.rerank_score,
            hybrid_score=metadata.get("hybrid_score", 0.0),
        )

        evidence_nodes.append(node)
        graph_state.nodes.append(node)
        graph.add_node(node.node_id, node=node)
        graph_state.node_lookup[node.node_id] = node

    if not evidence_nodes:
        raise ValueError(
            "Evidence Graph construction produced zero nodes."
        )

    # ======================================================
    # Build Semantic Edges
    # ======================================================
    edges = build_edges(
        evidence_nodes,
        similarity_threshold=0.55,
    ) or []

    for edge in edges:
        graph.add_edge(
            edge.source,
            edge.target,
            weight=edge.weight,
            relation=edge.relation,
        )
        graph_state.edges.append(edge)
        graph_state.edge_lookup[(edge.source, edge.target)] = edge

    graph_state.graph = graph

    # ======================================================
    # Basic Graph Statistics
    # ======================================================
    graph_state.statistics = compute_graph_statistics(graph)

    graph_state.signals.graph.node_count = graph.number_of_nodes()
    graph_state.signals.graph.edge_count = graph.number_of_edges()
    graph_state.signals.graph.graph_density = graph_state.statistics.graph_density
    graph_state.signals.graph.average_degree = graph_state.statistics.average_degree
    graph_state.signals.graph.connected_components = (
        graph_state.statistics.connected_components
    )

    # ======================================================
    # Clusters + Coherence
    # ======================================================
    graph_state = build_clusters(graph_state)
    graph_state = compute_graph_coherence(graph_state)

    # ======================================================
    # Optimization #34: lazy expensive analytics
    # ======================================================
    if analytics_mode == "lazy":
        sufficient, reason = _cheap_sufficiency_check(
            retrieval_state,
            graph_state,
        )
        graph_state.analytics_sufficiency = sufficient
        graph_state.analytics_skip_reason = reason

        if sufficient:
            graph_state.validation = validate_graph(graph_state)
            return graph_state

        # Ambiguous evidence falls through to the full analytics pipeline.
        return _compute_full_analytics(
            retrieval_state,
            graph_state,
        )

    # Historical/full mode.
    graph_state.analytics_sufficiency = False
    graph_state.analytics_skip_reason = "full_mode_requested"
    return _compute_full_analytics(
        retrieval_state,
        graph_state,
    )
