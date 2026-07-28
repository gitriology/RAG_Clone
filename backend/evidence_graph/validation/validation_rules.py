from backend.evidence_graph.validation.validation_utils import (
    add_error,
    add_warning,
    add_pass,
)


# ==========================================================
# NODE VALIDATION
# ==========================================================

def validate_nodes(
    graph_state,
    report,
):

    ids = set()

    for node in graph_state.nodes:

        if node.node_id in ids:

            add_error(
                report,
                f"Duplicate node id: {node.node_id}",
            )

        ids.add(node.node_id)

    add_pass(report)


# ==========================================================
# EDGE VALIDATION
# ==========================================================

def validate_edges(
    graph_state,
    report,
):

    node_ids = {

        n.node_id

        for n in graph_state.nodes

    }

    for edge in graph_state.edges:

        if edge.source not in node_ids:

            add_error(

                report,

                f"Missing edge source {edge.source}",

            )

        if edge.target not in node_ids:

            add_error(

                report,

                f"Missing edge target {edge.target}",

            )

    add_pass(report)


# ==========================================================
# CLUSTERS
# ==========================================================

def validate_clusters(
    graph_state,
    report,
):

    seen = set()

    for cluster in graph_state.clusters:

        if cluster.cluster_id in seen:

            add_error(

                report,

                "Duplicate cluster id",

            )

        seen.add(cluster.cluster_id)

    add_pass(report)


# ==========================================================
# SIGNALS
# ==========================================================

def validate_signals(
    graph_state,
    report,
):

    if not (0 <= graph_state.graph_score <= 1):

        add_warning(

            report,

            "Graph score outside [0,1]",

        )

    if not (0 <= graph_state.signals.coherence.score <= 1):

        add_warning(

            report,

            "Coherence outside [0,1]",

        )

    add_pass(report)


# ==========================================================
# RANKING
# ==========================================================

def validate_ranking(
    graph_state,
    report,
):

    if graph_state.ranking is None:

        add_warning(

            report,

            "Ranking not computed.",

        )

        return

    if len(graph_state.ranking.ranked_nodes) != len(graph_state.nodes):

        add_error(

            report,

            "Ranking size mismatch.",

        )

    add_pass(report)