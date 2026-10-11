from backend.ms_arc.state.retrieval_state import (
    RetrievalState,
)

from backend.ms_arc.state.retrieval_signals import (
    MarginMetrics,
)

from backend.models.model_registry import ModelRegistry
import math
from time import perf_counter


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

    rerank_started = perf_counter()
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

    # CrossEncoder outputs are ranking logits, not probabilities. A ratio
    # against abs(top_score) is unstable near zero and score magnitudes can be
    # negative. Map the non-negative logit gap through a saturating transform:
    # 1 - exp(-gap). Small gaps stay small; large gaps approach 1.
    normalized_margin = -math.expm1(-max(0.0, raw_margin))
    normalized_margin = max(0.0, min(1.0, normalized_margin))

    state.signals.margin = MarginMetrics(
        top_score=top_score,
        second_score=second_score,
        raw_margin=raw_margin,
        normalized_margin=normalized_margin,
    )

    state.debug["rerank_new_scores"] = len(missing_pairs)
    state.debug["rerank_reused_scores"] = len(state.selected_documents) - len(missing_pairs)
    state.debug["rerank_total_cached_scores"] = len(score_cache)
    state.debug["reranking_stage_seconds"] = perf_counter() - rerank_started
    state.debug["reranking_candidate_count"] = len(state.selected_documents)

    return state
