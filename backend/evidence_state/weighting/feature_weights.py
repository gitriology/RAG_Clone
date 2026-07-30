# ==========================================================
# FEATURE WEIGHTS
# ==========================================================

FEATURE_WEIGHTS = {

    # ---------------------------------------------
    # Retrieval
    # ---------------------------------------------

    "retrieval_confidence": 1.20,

    "retrieval_margin": 1.15,

    "retrieval_agreement": 1.10,

    "retrieval_complexity": 0.90,

    # ---------------------------------------------
    # Graph
    # ---------------------------------------------

    "graph_score": 1.25,

    "graph_density": 0.90,

    "graph_coherence": 1.20,

    "graph_consensus": 1.15,

    # ---------------------------------------------
    # Centrality
    # ---------------------------------------------

    "average_degree": 0.80,

    "average_betweenness": 0.90,

    "average_closeness": 0.90,

    "average_eigenvector": 1.00,

    "average_pagerank": 1.10,

    # ---------------------------------------------
    # Quality
    # ---------------------------------------------

    "evidence_quality": 1.30,

    # ---------------------------------------------
    # Diagnostics
    # ---------------------------------------------

    "vector_health": 1.00,

    "uncertainty": 1.10,

}


# ==========================================================
# APPLY FEATURE WEIGHTS
# ==========================================================

def apply_feature_weights(
    evidence_state,
):

    for feature in evidence_state.features:

        feature.weight = FEATURE_WEIGHTS.get(

            feature.name,

            1.0,

        )

    return evidence_state