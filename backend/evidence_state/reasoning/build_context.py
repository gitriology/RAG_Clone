# ==========================================================
# BUILD REASONING CONTEXT
# ==========================================================

def build_context(
    retrieval_state,
    graph_state,
    evidence_state,
    reasoning_state,
):
    """
    Collects structured context required by
    the Phase-10 reasoning engine.
    """

    context = reasoning_state.context

    context.query = retrieval_state.query

    context.document_count = len(
        retrieval_state.reranked_results
    )

    context.graph_nodes = len(
        graph_state.nodes
    )

    context.graph_edges = len(
        graph_state.edges
    )

    context.evidence_score = (
        evidence_state.evidence_score
    )

    return reasoning_state