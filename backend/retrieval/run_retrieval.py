from backend.ms_arc.run_msarc import (
    run_msarc
)


def retrieve(query):

    state = run_msarc(
        query
    )

    # ======================================================
    # OPTIMIZATION #4
    # ======================================================
    #
    # MS-ARC already executed the CrossEncoder and stored
    # reranked results in state.reranked_results.
    #
    # No second CrossEncoder inference occurs here.
    # ======================================================

    documents = []

    documents_source = (

        state.reranked_results

        if state.reranked_results

        else state.selected_documents

    )

    print(
        "[Optimization #4] "
        "Using MS-ARC reranked results:",
        bool(
            state.reranked_results
        )
    )

    for doc in documents_source:

        documents.append({

            "text":
                doc.text,

            "domain":
                doc.metadata.get(
                    "domain",
                    "general"
                ),

            "source":
                doc.metadata.get(
                    "source",
                    "unknown"
                ),

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
                        0.0
                    )
                ),

            "fusion_method":
                doc.metadata.get(
                    "fusion_method",
                    state.debug.get(
                        "fusion_method",
                        "minmax"
                    )
                ),

        })

    # ======================================================
    # OPTIMIZATION #6 EVIDENCE
    # ======================================================

    evidence = {

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
        # Optimization #6
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
                False
            ),

        "candidate_depth_delta":
            state.debug.get(
                "candidate_depth_delta",
                0
            ),

        "fusion_method":
            state.debug.get(
                "fusion_method",
                "minmax"
            ),

    }

    return {

        "documents":
            documents,

        "retrieval_confidence":
            state.retrieval_confidence,

        "evidence":
            evidence,

    }