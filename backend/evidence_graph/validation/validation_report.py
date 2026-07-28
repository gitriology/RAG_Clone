from dataclasses import dataclass, field
from typing import List


# ==========================================================
# VALIDATION REPORT
# ==========================================================

@dataclass
class GraphValidationReport:

    valid: bool = True

    errors: List[str] = field(
        default_factory=list
    )

    warnings: List[str] = field(
        default_factory=list
    )

    passed_checks: int = 0

    failed_checks: int = 0

    warning_checks: int = 0

    validation_score: float = 1.0