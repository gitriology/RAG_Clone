def rerank(query, documents):
    """
    Downstream reranking stage.

    Optimization #4
    ----------------
    CrossEncoder inference is performed once inside
    MS-ARC margin.py.

    This function MUST NOT call model.predict()
    again.

    It only reuses the rerank_score already attached
    to each document by MS-ARC.
    """

    # --------------------------------------------------
    # Safety check
    # --------------------------------------------------

    if not documents:
        return []

    # --------------------------------------------------
    # Verify that MS-ARC already supplied scores
    # --------------------------------------------------

    print(
        "[Optimization #4] Reusing existing CrossEncoder scores"
    )

    missing_scores = [
        doc
        for doc in documents
        if "rerank_score" not in doc
    ]

    if missing_scores:

        raise ValueError(
            "Optimization #4 error: "
            "rerank_score is missing from one or more "
            "documents. MS-ARC CrossEncoder results "
            "must be propagated before calling rerank()."
        )

    # --------------------------------------------------
    # Reuse existing CrossEncoder scores
    # --------------------------------------------------

    ranked = sorted(
        documents,
        key=lambda doc: float(
            doc.get(
                "rerank_score",
                0.0,
            )
        ),
        reverse=True,
    )

    return ranked