from backend.evidence_graph.state.evidence_graph_state import (
    EvidenceGraphState,
)

from backend.evidence_graph.models.graph_ranking_state import (
    GraphRankingState,
    RankedEvidence,
)

from backend.evidence_graph.ranking.ranking_score import (
    compute_node_graph_score,
)

from backend.evidence_graph.ranking.graph_consensus import (
    compute_consensus,
)


def build_graph_ranking(
    graph_state: EvidenceGraphState,
) -> GraphRankingState:
    """
    Phase 8E

    Produces the final graph-aware ranking using only
    EvidenceGraphState.

    RetrievalState is intentionally NOT required anymore.
    """

    ranking = GraphRankingState()

    coherence = graph_state.signals.coherence.score

    # --------------------------------------------------
    # Compute graph score for every evidence node
    # --------------------------------------------------

    for node in graph_state.nodes:

        score = compute_node_graph_score(
            node=node,
            coherence=coherence,
        )

        ranking.ranked_nodes.append(

            RankedEvidence(

                node=node,

                graph_score=score,

            )

        )

    # --------------------------------------------------
    # Sort
    # --------------------------------------------------

    ranking.ranked_nodes.sort(

        key=lambda x: x.graph_score,

        reverse=True,

    )

    # --------------------------------------------------
    # Assign ranks
    # --------------------------------------------------

    for rank, ranked in enumerate(

        ranking.ranked_nodes,

        start=1,

    ):

        ranked.rank = rank

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    if ranking.ranked_nodes:

        scores = [

            item.graph_score

            for item in ranking.ranked_nodes

        ]

        ranking.statistics.average_score = (

            sum(scores) / len(scores)

        )

        ranking.statistics.highest_score = max(scores)

        ranking.statistics.lowest_score = min(scores)

    # --------------------------------------------------
    # Consensus
    # --------------------------------------------------

    ranking.consensus = compute_consensus(

        graph_state.nodes,

        ranking.ranked_nodes,

    )

    return ranking