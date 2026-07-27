from backend.ms_arc.state.retrieval_state import (
    RetrievalState,
)

from backend.ms_arc.state.retrieval_signals import (
    MarginMetrics,
)

from backend.reranker.model.reranker import model


def compute_margin(
    state: RetrievalState,
) -> RetrievalState:
    """
    Computes CrossEncoder-based confidence margin.

    Produces

    • reranked_results
    • MarginMetrics
    """

    if not state.selected_documents:
        return state

    # =====================================================
    # Query-Document Pairs
    # =====================================================

    pairs = [

        (state.query, doc.text)

        for doc in state.selected_documents

    ]

    # =====================================================
    # CrossEncoder Scores
    # =====================================================

    scores = model.predict(pairs)

    # =====================================================
    # Attach Scores
    # =====================================================

    reranked = []

    for doc, score in zip(
        state.selected_documents,
        scores,
    ):

        doc.rerank_score = float(score)

        reranked.append(doc)

    # =====================================================
    # Sort by Relevance
    # =====================================================

    reranked.sort(
        key=lambda d: d.rerank_score,
        reverse=True,
    )

    state.reranked_results = reranked

    # =====================================================
    # Margin Computation
    # =====================================================

    top_score = reranked[0].rerank_score

    if len(reranked) > 1:
        second_score = reranked[1].rerank_score
    else:
        second_score = top_score

    raw_margin = top_score - second_score

    epsilon = 1e-8

    normalized_margin = raw_margin / (
        abs(top_score) + epsilon
    )

    # =====================================================
    # Typed Metrics
    # =====================================================

    state.signals.margin = MarginMetrics(

        top_score=top_score,

        second_score=second_score,

        raw_margin=raw_margin,

        normalized_margin=normalized_margin,

    )

    return state