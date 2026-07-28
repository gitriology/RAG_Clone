from backend.evidence_graph.validation.validation_report import (
    GraphValidationReport,
)


def add_error(
    report: GraphValidationReport,
    message: str,
):

    report.valid = False

    report.errors.append(message)

    report.failed_checks += 1


def add_warning(
    report: GraphValidationReport,
    message: str,
):

    report.warnings.append(message)

    report.warning_checks += 1


def add_pass(
    report: GraphValidationReport,
):

    report.passed_checks += 1