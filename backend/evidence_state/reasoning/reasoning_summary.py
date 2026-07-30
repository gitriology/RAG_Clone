# ==========================================================
# BUILD SUMMARY
# ==========================================================

def build_summary(
    retrieval_state,
    graph_state,
    evidence_state,
    reasoning_state,
):
    """
    Produces a concise machine-readable summary.
    """

    summary = reasoning_state.summary

    summary.summary = (

        f"{len(retrieval_state.reranked_results)} evidence "

        f"documents, "

        f"{len(graph_state.nodes)} graph nodes, "

        f"Evidence Score "

        f"{evidence_state.evidence_score:.3f}"

    )

    return reasoning_state