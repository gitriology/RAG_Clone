"""Compatibility wrapper for the shared production answerability gate."""

from __future__ import annotations

from typing import Any, Dict, Iterable

from backend.retrieval.query_matching import assess_answerability


def assess_selected_evidence(
    query: str,
    evidence: Iterable[Dict[str, Any]] | None,
) -> Dict[str, Any]:
    """Return the canonical query/evidence answerability assessment."""
    return assess_answerability(query, evidence or [])
