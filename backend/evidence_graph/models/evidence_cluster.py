from dataclasses import dataclass, field
from typing import List

# ==========================================================
# EVIDENCE CLUSTER
# ==========================================================

@dataclass
class EvidenceCluster:

    cluster_id: int

    node_ids: List[str] = field(default_factory=list)

    centroid_node: str = ""

    score: float = 0.0