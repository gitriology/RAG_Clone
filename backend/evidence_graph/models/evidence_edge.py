from dataclasses import dataclass

# ==========================================================
# EVIDENCE EDGE
# ==========================================================

@dataclass
class EvidenceEdge:

    source: str

    target: str

    weight: float

    relation: str