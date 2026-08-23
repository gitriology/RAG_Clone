"""
Adaptive Candidate Pool Controller for MS-ARC.

Optimization #6
----------------
Determines retrieval candidate depth dynamically using:

1. Query complexity
2. Query type
3. Retrieval disagreement
4. Retrieval stability
5. Short entity / acronym characteristics

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
    """
    Normalize query type.
    """

    if query_type is None:
        return ""

    return str(query_type).strip().lower()


# ==========================================================
# SAFE SIGNAL NORMALIZATION
# ==========================================================

def _clip_signal(value, default=0.5):
    """
    Convert a signal into [0, 1].
    """

    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default

    return max(0.0, min(1.0, value))


# ==========================================================
# SHORT ENTITY / ACRONYM DETECTION
# ==========================================================

def is_short_entity_query(query: str) -> bool:
    """
    Detect short factual/entity-style queries.

    Examples:
        What is WHO?
        What is NASA?
        What is ISRO?
        Define AI
        What is FAISS?
    """

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

    # ------------------------------------------------------
    # Explicit acronym/entity detection
    # ------------------------------------------------------

    for word in words:

        cleaned = word.strip(
            ".,?!:;()[]{}\"'"
        )

        if (
            2 <= len(cleaned) <= 8
            and cleaned.isupper()
        ):
            return True

    # ------------------------------------------------------
    # A short definition-style query is still retrieval
    # sensitive even if the entity is not uppercase.
    # ------------------------------------------------------

    return True


# ==========================================================
# COMPLEXITY CONTRIBUTION
# ==========================================================

def _complexity_contribution(complexity: float) -> int:
    """
    Convert query complexity into candidate-depth
    contribution.

    Complexity is interpreted continuously rather than
    relying only on query_type.
    """

    if complexity < 0.30:
        return 0

    if complexity < 0.55:
        return 10

    if complexity < 0.75:
        return 20

    return 30


# ==========================================================
# DISAGREEMENT CONTRIBUTION
# ==========================================================

def _disagreement_contribution(agreement: float) -> int:
    """
    Low dense/BM25 agreement means high retrieval
    uncertainty.

    Therefore:

        high agreement -> no expansion
        medium agreement -> small expansion
        low agreement -> large expansion
    """

    if agreement < 0.20:
        return 10

    if agreement < 0.40:
        return 5

    return 0


# ==========================================================
# INSTABILITY CONTRIBUTION
# ==========================================================

def _instability_contribution(stability: float) -> int:
    """
    Low stability means retrieval ranking is sensitive
    to Top-K changes.

    Therefore a wider candidate pool is useful.
    """

    if stability < 0.40:
        return 10

    if stability < 0.60:
        return 5

    return 0


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

        Query complexity
        +
        Retrieval disagreement
        +
        Retrieval instability
        +
        Entity sensitivity

    Parameters
    ----------
    query_complexity:
        Complexity score in [0, 1].

    query_type:
        simple / medium / complex / very_complex.

    agreement:
        Dense/BM25 retrieval agreement in [0, 1].

    stability:
        Retrieval stability in [0, 1].

    requested_k:
        Final number of documents required.

    query:
        Original user query.

    Returns
    -------
    int
        Adaptive candidate pool size.
    """

    # ======================================================
    # Normalize signals
    # ======================================================

    complexity = _clip_signal(
        query_complexity,
        default=0.0
    )

    agreement = _clip_signal(
        agreement,
        default=0.5
    )

    stability = _clip_signal(
        stability,
        default=0.5
    )

    query_type = _normalize_query_type(
        query_type
    )

    # ======================================================
    # Determine requested K
    # ======================================================

    if requested_k is None:

        requested_k = 3

    try:
        requested_k = int(
            requested_k
        )

    except (TypeError, ValueError):

        requested_k = 3

    requested_k = max(
        1,
        requested_k
    )

    # ======================================================
    # BASE DEPTH
    # ======================================================
    #
    # The baseline is deliberately proportional to the
    # requested result size rather than:
    #
    #     max(k * 4, 20)
    #
    # used by the old implementation.
    #
    # MS-ARC then expands this according to uncertainty.
    # ======================================================

    base_candidate_k = max(
        requested_k * 2,
        MIN_CANDIDATE_K
    )

    # ======================================================
    # Complexity contribution
    # ======================================================

    complexity_addition = (
        _complexity_contribution(
            complexity
        )
    )

    # ======================================================
    # Query-type contribution
    # ======================================================

    query_type_addition = 0

    if query_type == "medium":
        query_type_addition = 5

    elif query_type == "complex":
        query_type_addition = 10

    elif query_type == "very_complex":
        query_type_addition = 15

    # Avoid double-counting complexity too aggressively.
    #
    # Query type is used only as a small refinement.
    query_type_addition = min(
        query_type_addition,
        10
    )

    # ======================================================
    # Short entity contribution
    # ======================================================

    short_entity = is_short_entity_query(
        query
    )

    entity_addition = 5 if short_entity else 0

    # ======================================================
    # Retrieval disagreement
    # ======================================================

    disagreement_addition = (
        _disagreement_contribution(
            agreement
        )
    )

    # ======================================================
    # Retrieval instability
    # ======================================================

    instability_addition = (
        _instability_contribution(
            stability
        )
    )

    # ======================================================
    # Combine controller signals
    # ======================================================

    candidate_k = (

        base_candidate_k

        + complexity_addition

        + query_type_addition

        + entity_addition

        + disagreement_addition

        + instability_addition

    )

    # ======================================================
    # Ensure enough candidates
    # ======================================================

    candidate_k = max(
        candidate_k,
        requested_k
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
        f"Query Type              : {query_type}"
    )

    print(
        f"Query Complexity        : {complexity:.3f}"
    )

    print(
        f"Short Entity            : {short_entity}"
    )

    print(
        f"Agreement               : {agreement:.3f}"
    )

    print(
        f"Stability               : {stability:.3f}"
    )

    print(
        f"Requested Top-K         : {requested_k}"
    )

    print(
        f"Base Candidate K        : {base_candidate_k}"
    )

    print(
        f"Complexity Addition     : "
        f"{complexity_addition}"
    )

    print(
        f"Query-Type Addition     : "
        f"{query_type_addition}"
    )

    print(
        f"Entity Addition         : "
        f"{entity_addition}"
    )

    print(
        f"Disagreement Addition   : "
        f"{disagreement_addition}"
    )

    print(
        f"Instability Addition    : "
        f"{instability_addition}"
    )

    print(
        f"Final Candidate K       : "
        f"{candidate_k}"
    )

    print()

    return int(candidate_k)