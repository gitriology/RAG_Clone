from backend.evidence_graph.validation.validation_report import (
    GraphValidationReport,
)

from backend.evidence_graph.validation.validation_rules import (

    validate_nodes,

    validate_edges,

    validate_clusters,

    validate_signals,

    validate_ranking,

)


def validate_graph(
    graph_state,
):
    """
    Complete structural validation
    of the Evidence Graph.
    """

    report = GraphValidationReport()

    validate_nodes(
        graph_state,
        report,
    )

    validate_edges(
        graph_state,
        report,
    )

    validate_clusters(
        graph_state,
        report,
    )

    validate_signals(
        graph_state,
        report,
    )

    validate_ranking(
        graph_state,
        report,
    )

    total = (

        report.passed_checks +

        report.failed_checks +

        report.warning_checks

    )

    if total:

        report.validation_score = (

            report.passed_checks / total

        )

    return report