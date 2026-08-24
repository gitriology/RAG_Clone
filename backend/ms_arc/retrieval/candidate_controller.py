"""
Adaptive Candidate Pool Controller for MS-ARC.

Optimization #6
----------------
Determines retrieval candidate depth dynamically using:

1. Query complexity
2. Query type
3. Dense/Sparse disagreement
4. Retrieval stability
5. Short entity/acronym characteristics

The controller ONLY determines candidate depth.

It does NOT perform retrieval.
"""

from typing import Optional


# ==========================================================
# CONFIGURATION
# ==========================================================

MIN_CANDIDATE_K = 10
MAX_CANDIDATE_K = 40


# ==========================================================
# QUERY TYPE NORMALIZATION
# ==========================================================

def _normalize_query_type(query_type: str) -> str:

    if query_type is None:
        return ""

    return str(query_type).strip().lower()


# ==========================================================
# SHORT ENTITY / ACRONYM DETECTION
# ==========================================================

def is_short_entity_query(query: str) -> bool:

    if not query:
        return False

    normalized = " ".join(
        str(query).strip().split()
    )

    words = normalized.split()

    if len(words) > 8:
        return False

    lowered = normalized.lower()

    definition_patterns = (
        "what is ",
        "what are ",
        "who is ",
        "define ",
        "meaning of ",
    )

    is_definition_style = any(
        lowered.startswith(pattern)
        for pattern in definition_patterns
    )

    if not is_definition_style:
        return False

    for word in words:

        cleaned = word.strip(
            ".,?!:;()[]{}\"'"
        )

        if (
            2 <= len(cleaned) <= 8
            and cleaned.isupper()
        ):
            return True

    # Short definition questions are still
    # retrieval-sensitive.
    return False


# ==========================================================
# SIGNAL NORMALIZATION
# ==========================================================

def _clamp(value: float) -> float:

    return max(
        0.0,
        min(1.0, value)
    )


# ==========================================================
# CANDIDATE CONTROLLER
# ==========================================================

def determine_candidate_k(
    query_complexity: float = 0.0,
    query_type: str = "",
    agreement: float = 0.5,
    stability: float = 0.5,
    requested_k: Optional[int] = None,
    query: str = "",
) -> int:
    """
    Determine adaptive candidate pool size.

    The controller combines:

        complexity
        + disagreement
        + instability
        + short-entity sensitivity

    Lower agreement/stability means greater uncertainty,
    therefore a larger candidate pool.

    Returns
    -------
    int
        Adaptive candidate pool size.
    """

    # ======================================================
    # Normalize complexity
    # ======================================================

    try:
        complexity = float(
            query_complexity
        )
    except (
        TypeError,
        ValueError,
    ):
        complexity = 0.0

    complexity = _clamp(
        complexity
    )

    # ======================================================
    # Normalize agreement
    # ======================================================

    try:
        agreement = float(
            agreement
        )
    except (
        TypeError,
        ValueError,
    ):
        agreement = 0.5

    agreement = _clamp(
        agreement
    )

    # ======================================================
    # Normalize stability
    # ======================================================

    try:
        stability = float(
            stability
        )
    except (
        TypeError,
        ValueError,
    ):
        stability = 0.5

    stability = _clamp(
        stability
    )

    query_type = _normalize_query_type(
        query_type
    )

    # ======================================================
    # Base candidate pool
    # ======================================================

    if query_type == "simple":

        candidate_k = 10

    elif query_type == "medium":

        candidate_k = 20

    elif query_type == "complex":

        candidate_k = 30

    elif query_type == "very_complex":

        candidate_k = 40

    else:

        if complexity < 0.30:
            candidate_k = 10

        elif complexity < 0.55:
            candidate_k = 20

        elif complexity < 0.75:
            candidate_k = 30

        else:
            candidate_k = 40

    base_candidate_k = candidate_k

    # ======================================================
    # Short entity / acronym
    # ======================================================

    short_entity = is_short_entity_query(
        query
    )

    if short_entity:

        candidate_k = max(
            candidate_k,
            15
        )

    # ======================================================
    # Retrieval disagreement
    # ======================================================

    disagreement = 1.0 - agreement

    if disagreement >= 0.75:

        candidate_k += 10

    elif disagreement >= 0.50:

        candidate_k += 5

    # ======================================================
    # Retrieval instability
    # ======================================================

    instability = 1.0 - stability

    if instability >= 0.60:

        candidate_k += 10

    elif instability >= 0.40:

        candidate_k += 5

    # ======================================================
    # Requested K constraint
    # ======================================================

    requested_k_value = None

    if requested_k is not None:

        try:
            requested_k_value = int(
                requested_k
            )
        except (
            TypeError,
            ValueError,
        ):
            requested_k_value = 1

        requested_k_value = max(
            1,
            requested_k_value
        )

        candidate_k = max(
            candidate_k,
            requested_k_value * 2
        )

    # ======================================================
    # Clamp
    # ======================================================

    candidate_k = max(
        MIN_CANDIDATE_K,
        min(
            MAX_CANDIDATE_K,
            candidate_k
        )
    )

    # ======================================================
    # Debug
    # ======================================================

    print()
    print(
        "[Optimization #6] Candidate Controller"
    )

    print(
        f"Query Type        : {query_type}"
    )

    print(
        f"Query Complexity  : {complexity:.3f}"
    )

    print(
        f"Short Entity      : {short_entity}"
    )

    print(
        f"Agreement         : {agreement:.3f}"
    )

    print(
        f"Disagreement      : {disagreement:.3f}"
    )

    print(
        f"Stability         : {stability:.3f}"
    )

    print(
        f"Instability       : {instability:.3f}"
    )

    print(
        f"Base Candidate K  : {base_candidate_k}"
    )

    print(
        f"Final Candidate K : {candidate_k}"
    )

    print()

    return int(candidate_k)