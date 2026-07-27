from backend.ms_arc.state.retrieval_state import RetrievalState
from backend.ms_arc.state.retrieval_signals import AgreementMetrics


def compute_agreement(
    state: RetrievalState,
) -> RetrievalState:
    """
    Computes the weighted agreement between
    Dense Retrieval and Sparse Retrieval.

    Agreement =
        Sum(intersection weights)
        -------------------------
           Sum(union weights)

    where

        weight = dense_score + sparse_score
    """

    # =====================================================
    # Dense Lookup
    # =====================================================

    dense = {
        doc.doc_id: doc
        for doc in state.dense_results
    }

    # =====================================================
    # Sparse Lookup
    # =====================================================

    sparse = {
        doc.doc_id: doc
        for doc in state.sparse_results
    }

    # =====================================================
    # Union / Intersection
    # =====================================================

    dense_ids = set(dense.keys())

    sparse_ids = set(sparse.keys())

    intersection = dense_ids & sparse_ids

    union = dense_ids | sparse_ids

    # =====================================================
    # Weighted Intersection
    # =====================================================

    numerator = 0.0

    for doc_id in intersection:

        numerator += (

            dense[doc_id].dense_score +

            sparse[doc_id].sparse_score

        )

    # =====================================================
    # Weighted Union
    # =====================================================

    denominator = 0.0

    for doc_id in union:

        weight = 0.0

        if doc_id in dense:
            weight += dense[doc_id].dense_score

        if doc_id in sparse:
            weight += sparse[doc_id].sparse_score

        denominator += weight

    # =====================================================
    # Agreement Score
    # =====================================================

    if denominator == 0:

        agreement = 0.0

    else:

        agreement = numerator / denominator

    # =====================================================
    # Typed Metrics
    # =====================================================

    state.signals.agreement = AgreementMetrics(

        score=agreement,

        intersection=len(intersection),

        union=len(union),

    )

    return state