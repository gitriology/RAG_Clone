from backend.ms_arc.run_msarc import (
    run_msarc,
)


def retrieve(
    query,
    fusion_method="minmax",
):
    """
    Execute MS-ARC retrieval and expose the
    document IDs required by evaluation.

    Optimization #4:
        Reuse MS-ARC reranked results.

    Optimization #6:
        Preserve the selected fusion method and
        expose candidate-controller evidence.
    """

    fusion_method = (
        fusion_method
        or "minmax"
    ).lower().strip()

    if fusion_method not in {
        "minmax",
        "rrf",
    }:

        raise ValueError(
            f"Unsupported fusion method: "
            f"{fusion_method}"
        )

    # ======================================================
    # MS-ARC
    # ======================================================

    state = run_msarc(

        query,

        fusion_method=fusion_method,

    )

    # ======================================================
    # Optimization #4
    # ======================================================
    #
    # MS-ARC already executed the CrossEncoder and stored
    # reranked results in state.reranked_results.
    #
    # Reuse them instead of running CrossEncoder again.
    # ======================================================

    if state.reranked_results:

        documents_source = (
            state.reranked_results
        )

        using_reranked = True

    else:

        documents_source = (
            state.selected_documents
        )

        using_reranked = False

    print(
        "[Optimization #4] "
        "Using MS-ARC reranked results:",
        using_reranked,
    )

    # ======================================================
    # BUILD DOCUMENT OUTPUT
    # ======================================================

    documents = []

    for doc in documents_source:

        documents.append({

            # ------------------------------------------------
            # CRITICAL FOR EVALUATION
            # ------------------------------------------------

            "doc_id":
                int(doc.doc_id),

            # ------------------------------------------------
            # Content
            # ------------------------------------------------

            "text":
                doc.text,

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            "domain":
                doc.metadata.get(
                    "domain",
                    "general",
                ),

            "source":
                doc.metadata.get(
                    "source",
                    "unknown",
                ),

            # ------------------------------------------------
            # Scores
            # ------------------------------------------------

            "rerank_score":
                float(
                    doc.rerank_score
                ),

            "dense_score":
                float(
                    doc.dense_score
                ),

            "sparse_score":
                float(
                    doc.sparse_score
                ),

            "hybrid_score":
                float(
                    doc.metadata.get(
                        "hybrid_score",
                        0.0,
                    )
                ),

            # ------------------------------------------------
            # Fusion
            # ------------------------------------------------

            "fusion_method":
                doc.metadata.get(
                    "fusion_method",
                    fusion_method,
                ),

            # ------------------------------------------------
            # Ranking diagnostics
            # ------------------------------------------------

            "dense_rank":
                doc.metadata.get(
                    "dense_rank",
                ),

            "sparse_rank":
                doc.metadata.get(
                    "sparse_rank",
                ),

        })

    # ======================================================
    # OPTIMIZATION #6 EVIDENCE
    # ======================================================

    evidence = {

        # --------------------------------------------------
        # Retrieval signals
        # --------------------------------------------------

        "agreement":
            state.signals.agreement.score,

        "margin":
            state.signals.margin.normalized_margin,

        "stability":
            state.signals.stability.score,

        "decision":
            state.signals.decision.decision,

        "reason":
            state.signals.decision.reason,

        # --------------------------------------------------
        # Candidate controller
        # --------------------------------------------------

        "initial_candidate_k":
            state.debug.get(
                "initial_candidate_k"
            ),

        "adaptive_candidate_k":
            state.debug.get(
                "adaptive_candidate_k"
            ),

        "final_candidate_k":
            state.debug.get(
                "final_candidate_k"
            ),

        "candidate_expanded":
            state.debug.get(
                "candidate_expanded",
                False,
            ),

        "candidate_depth_delta":
            state.debug.get(
                "candidate_depth_delta",
                0,
            ),

        "candidate_k_used":
            state.debug.get(
                "candidate_k_used"
            ),

        # --------------------------------------------------
        # Fusion
        # --------------------------------------------------

        "fusion_method":
            state.debug.get(
                "fusion_method",
                fusion_method,
            ),

        "stability_fusion_method":
            state.debug.get(
                "stability_fusion_method",
                fusion_method,
            ),

    }

    # ======================================================
    # FINAL RESULT
    # ======================================================

    return {

        "documents":
            documents,

        "retrieval_confidence":
            state.retrieval_confidence,

        "evidence":
            evidence,

    }