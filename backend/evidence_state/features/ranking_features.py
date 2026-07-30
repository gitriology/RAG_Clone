from backend.evidence_state.state.evidence_state import (
    EvidenceFeature,
)


def extract_ranking_features(
    graph_state,
):
    """
    Graph Ranking Features
    """

    ranking = graph_state.signals.ranking

    features = [

        EvidenceFeature(
            "ranking_average_score",
            ranking.average_score,
            "ranking",
        ),

        EvidenceFeature(
            "ranking_highest_score",
            ranking.highest_score,
            "ranking",
        ),

        EvidenceFeature(
            "ranking_lowest_score",
            ranking.lowest_score,
            "ranking",
        ),

    ]

    return features