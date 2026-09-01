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

    Optimization #21:
        Preserve the complete RetrievalState so downstream
        production stages can construct the Evidence Graph
        and Evidence State from the same MS-ARC execution.

    Returns
    -------
    dict
        Public retrieval result containing:

        - documents
        - retrieval_confidence
        - evidence
        - retrieval_state
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

            "ranking_score":
                float(
                    doc.metadata.get(
                        "ranking_score",
                        doc.metadata.get(
                            "hybrid_score",
                            0.0,
                        ),
                    )
                ),

            "lexical_anchor_score":
                float(
                    doc.metadata.get(
                        "lexical_anchor_score",
                        0.0,
                    )
                ),

            "exact_phrase_match":
                bool(
                    doc.metadata.get(
                        "exact_phrase_match",
                        False,
                    )
                ),

            "reference_match":
                bool(
                    doc.metadata.get(
                        "reference_match",
                        False,
                    )
                ),

            "matched_focus":
                list(
                    doc.metadata.get(
                        "matched_focus",
                        [],
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
    # Optimization #21
    # ======================================================
    #
    # Preserve the actual RetrievalState.
    #
    # This is intentionally an internal pipeline object.
    # The API layer will expose only serializable summaries.
    # ======================================================

    print()
    print("=" * 70)
    print(
        "OPTIMIZATION #21 : Retrieval State Preserved"
    )
    print("=" * 70)

    print(
        "RetrievalState available      : PASS"
    )

    print(
        "Reranked documents available  : "
        f"{len(state.reranked_results)}"
    )

    print(
        "Selected documents available  : "
        f"{len(state.selected_documents)}"
    )

    print("=" * 70)

    # ======================================================
    # FINAL RESULT
    # ======================================================

    return {

        "documents":
            documents,

        "retrieval_confidence":
            float(
                state.retrieval_confidence
            ),

        "evidence":
            evidence,

        # --------------------------------------------------
        # Optimization #21
        # --------------------------------------------------
        #
        # Internal state used by:
        #
        #   Evidence Graph
        #   Evidence State
        #
        # --------------------------------------------------

        "retrieval_state":
            state,
    }