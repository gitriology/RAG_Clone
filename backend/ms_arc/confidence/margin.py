from backend.ms_arc.state.retrieval_state import (
    RetrievalState,
)

from backend.ms_arc.state.retrieval_signals import (
    MarginMetrics,
)

from backend.models.model_registry import ModelRegistry


def compute_margin(
    state: RetrievalState,
) -> RetrievalState:
    """Compute Cross-Encoder margin while reusing prior iteration scores."""

    if not state.selected_documents:
        return state

    score_cache = state.debug.setdefault(
        "rerank_score_cache",
        {},
    )

    missing_docs = []
    missing_pairs = []

    for doc in state.selected_documents:
        cached = score_cache.get(str(doc.doc_id))
        if cached is None:
            missing_docs.append(doc)
            missing_pairs.append((state.query, doc.text))
        else:
            doc.rerank_score = float(cached)

    if missing_pairs:
        model = ModelRegistry.get_reranker_model()

        print(
            "[Optimization #28] CrossEncoder scoring new documents:",
            len(missing_pairs),
        )

        scores = model.predict(missing_pairs)

        for doc, score in zip(missing_docs, scores):
            doc.rerank_score = float(score)
            score_cache[str(doc.doc_id)] = float(score)
    else:
        print(
            "[Optimization #28] CrossEncoder cache: all scores reused"
        )

    reranked = list(state.selected_documents)
    reranked.sort(
        key=lambda d: d.rerank_score,
        reverse=True,
    )
    state.reranked_results = reranked

    top_score = reranked[0].rerank_score
    second_score = (
        reranked[1].rerank_score
        if len(reranked) > 1
        else top_score
    )

    raw_margin = top_score - second_score
    epsilon = 1e-8
    normalized_margin = raw_margin / (abs(top_score) + epsilon)

    state.signals.margin = MarginMetrics(
        top_score=top_score,
        second_score=second_score,
        raw_margin=raw_margin,
        normalized_margin=normalized_margin,
    )

    state.debug["rerank_new_scores"] = len(missing_pairs)
    state.debug["rerank_reused_scores"] = len(state.selected_documents) - len(missing_pairs)
    state.debug["rerank_total_cached_scores"] = len(score_cache)

    return state
