from backend.evidence_state.state.evidence_state import (
    EvidenceFeature,
)


# ==========================================================
# RETRIEVAL FEATURES
# ==========================================================

def extract_retrieval_features(
    retrieval_state,
):
    """
    Converts the MS-ARC RetrievalState into
    Evidence Features for Phase 9.
    """

    signals = retrieval_state.signals

    features = [

        # ==================================================
        # Core Retrieval Signals
        # ==================================================

        EvidenceFeature(

            name="agreement",

            value=signals.agreement.score,

            source="retrieval",

            description="Dense/Sparse retrieval agreement",

        ),

        EvidenceFeature(

            name="margin",

            value=signals.margin.normalized_margin,

            source="retrieval",

            description="Confidence margin between top documents",

        ),

        EvidenceFeature(

            name="stability",

            value=signals.stability.score,

            source="retrieval",

            description="Retrieval stability",

        ),

        EvidenceFeature(

            name="retrieval_confidence",

            value=retrieval_state.retrieval_confidence,

            source="retrieval",

            description="Overall retrieval confidence",

        ),

        # ==================================================
        # Query Analysis
        # ==================================================

        EvidenceFeature(

            name="query_complexity",

            value=retrieval_state.query_complexity,

            source="retrieval",

            description="Estimated query complexity",

        ),

        EvidenceFeature(

            name="recommended_topk",

            value=float(

                retrieval_state.recommended_topk

            ),

            source="retrieval",

            description="Adaptive retrieval Top-K",

        ),

        # ==================================================
        # Novelty
        # ==================================================

        EvidenceFeature(

            name="novelty",

            value=signals.novelty.score,

            source="retrieval",

            description="Novel information retrieved",

        ),

        # ==================================================
        # Evidence Quality
        # ==================================================

        EvidenceFeature(

            name="evidence_quality",

            value=signals.evidence.score,

            source="retrieval",

            description="Overall evidence quality",

        ),

        EvidenceFeature(

            name="evidence_coverage",

            value=signals.evidence.coverage,

            source="retrieval",

            description="Evidence coverage",

        ),

        EvidenceFeature(

            name="evidence_diversity",

            value=signals.evidence.diversity,

            source="retrieval",

            description="Evidence diversity",

        ),

        EvidenceFeature(

            name="evidence_consistency",

            value=signals.evidence.consistency,

            source="retrieval",

            description="Evidence consistency",

        ),

    ]

    return features