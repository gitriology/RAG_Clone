"""Shared Cross-Encoder reranking utilities.

Optimization #29
----------------
The Cross-Encoder is owned by ``ModelRegistry`` and is therefore loaded once
per process.  MS-ARC performs the canonical scoring pass and stores
``rerank_score`` on the retrieved documents.  Downstream callers reuse those
scores instead of invoking ``predict`` a second time.
"""

def get_shared_reranker_model():
    """Return the process-shared Cross-Encoder instance."""
    from backend.models.model_registry import ModelRegistry

    return ModelRegistry.get_reranker_model()


def rerank(query, documents):
    """Sort documents using Cross-Encoder scores already computed by MS-ARC.

    ``query`` is retained for API compatibility.  It is intentionally not
    used to score documents here: MS-ARC is the canonical scoring stage.
    """
    del query

    if not documents:
        return []

    missing_scores = [
        doc for doc in documents
        if _get_score(doc) is None
    ]

    if missing_scores:
        raise ValueError(
            "Optimization #29 error: rerank_score is missing from one or "
            "more documents. MS-ARC must provide the canonical Cross-Encoder "
            "scores before downstream reranking."
        )

    return sorted(
        documents,
        key=lambda doc: float(_get_score(doc)),
        reverse=True,
    )


def _get_score(document):
    if isinstance(document, dict):
        return document.get("rerank_score")
    return getattr(document, "rerank_score", None)
