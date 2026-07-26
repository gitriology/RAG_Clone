from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RetrievedDocument:
    """
    Represents a single retrieved document along with all retrieval scores.
    """

    doc_id: str
    text: str

    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalState:
    """
    Shared state object passed through the complete MS-ARC pipeline.

    Every Phase 7 module updates this object instead of creating new variables.
    """

    # -------------------------
    # Query
    # -------------------------
    query: str

    query_type: Optional[str] = None

    query_complexity: float = 0.0

    recommended_topk: int = 5

    # -------------------------
    # Retrieval Signals
    # -------------------------

    agreement_score: float = 0.0

    confidence_margin: float = 0.0

    retrieval_stability: float = 0.0

    retrieval_confidence: float = 0.0

    # -------------------------
    # Decision
    # -------------------------

    decision: str = "Retrieve"

    # -------------------------
    # Documents
    # -------------------------

    dense_results: List[RetrievedDocument] = field(default_factory=list)

    sparse_results: List[RetrievedDocument] = field(default_factory=list)

    merged_results: List[RetrievedDocument] = field(default_factory=list)

    selected_documents: List[RetrievedDocument] = field(default_factory=list)

    # -------------------------
    # Future Phases
    # -------------------------

    evidence_graph = None

    evidence_state_vector = None

    reasoning_state = None

    uncertainty_score: float = 0.0

    debug: Dict[str, Any] = field(default_factory=dict)