from dataclasses import dataclass, field
from typing import List, Dict

from backend.ms_arc.state.retrieval_signals import RetrievalSignals


# ==========================================================
# DOCUMENT
# ==========================================================

@dataclass
class RetrievedDocument:

    doc_id: str

    text: str

    dense_score: float = 0.0

    sparse_score: float = 0.0

    rerank_score: float = 0.0

    metadata: Dict = field(
        default_factory=dict
    )


# ==========================================================
# RETRIEVAL STATE
# ==========================================================

@dataclass
class RetrievalState:

    # ======================================================
    # USER QUERY
    # ======================================================

    query: str

    # ======================================================
    # QUERY ANALYSIS
    # ======================================================

    query_complexity: float = 0.0

    query_type: str = ""

    recommended_topk: int = 5

    complexity_details: Dict = field(
        default_factory=dict
    )

    # ======================================================
    # RETRIEVAL RESULTS
    # ======================================================

    dense_results: List[RetrievedDocument] = field(
        default_factory=list
    )

    sparse_results: List[RetrievedDocument] = field(
        default_factory=list
    )

    merged_results: List[RetrievedDocument] = field(
        default_factory=list
    )

    selected_documents: List[RetrievedDocument] = field(
        default_factory=list
    )

    reranked_results: List[RetrievedDocument] = field(
        default_factory=list
    )

    # ======================================================
    # RETRIEVAL SIGNALS
    # ======================================================

    signals: RetrievalSignals = field(
        default_factory=RetrievalSignals
    )

    # ======================================================
    # FINAL OUTPUT
    # ======================================================

    retrieval_confidence: float = 0.0

    routing_domain: str = ""

    routing_confidence: float = 0.0

    decision: str = ""

    # ======================================================
    # DEBUG INFORMATION
    # ======================================================

    debug: Dict = field(
        default_factory=dict
    )