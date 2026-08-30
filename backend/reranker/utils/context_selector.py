"""
Context selection utility.

Ranks validated documents using both:
- CrossEncoder/MS-ARC rerank score
- context relevance

and preserves the existing select_top_k(ranked_docs, k)
API.
"""

from __future__ import annotations

import math


def _sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def select_top_k(ranked_docs, k=3):
    if not ranked_docs or k <= 0:
        return []

    docs = list(ranked_docs)

    for doc in docs:
        rerank_logit = float(
            doc.get("rerank_score", 0.0)
        )

        rerank_probability = _sigmoid(
            rerank_logit
        )

        context_score = float(
            doc.get("context_score", 0.0)
        )

        quality = float(
            doc.get("context_quality", 1.0)
        )

        sentence_support = float(
            doc.get("context_sentence_score", 0.0)
        )

        # Combined context ranking.
        #
        # Rerank remains the strongest signal, but a document now
        # needs actual semantic context support as well.
        combined = (
            0.45 * rerank_probability
            + 0.35 * context_score
            + 0.15 * sentence_support
            + 0.05 * quality
        )

        doc["_context_selection_score"] = combined

    docs.sort(
        key=lambda doc: (
            float(
                doc.get(
                    "_context_selection_score",
                    0.0,
                )
            ),
            float(
                doc.get(
                    "context_score",
                    0.0,
                )
            ),
            float(
                doc.get(
                    "rerank_score",
                    0.0,
                )
            ),
        ),
        reverse=True,
    )

    selected = docs[:k]

    for doc in selected:
        doc.pop(
            "_context_selection_score",
            None,
        )

    return selected
