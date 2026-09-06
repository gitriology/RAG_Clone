from dataclasses import dataclass, field
from typing import Any, List, Dict

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

    # Retrieval-stage EvidenceState snapshot used by Optimization #36.
    evidence_state: Any = None

    # Optimization #37: demand-driven full EvidenceState/Graph snapshots
    # created inside the adaptive controller when retrieval remains ambiguous.
    # These are reused by the downstream production pipeline on the final
    # iteration, avoiding a second graph/state construction.
    evidence_graph_state: Any = None
    controller_evidence_state: Any = None

    # ======================================================
    # DEBUG INFORMATION
    # ======================================================

    debug: Dict = field(
        default_factory=dict
    )