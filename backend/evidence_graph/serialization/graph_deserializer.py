import json

from backend.evidence_graph.serialization.graph_snapshot import (
    GraphSnapshot,
    SerializedNode,
    SerializedEdge,
)
from backend.evidence_graph.serialization.graph_snapshot import (
    SerializedGraphStatistics,
    SerializedGraphSignals,
)

# ==========================================================
# LOAD SNAPSHOT
# ==========================================================

def load_snapshot(
    filename,
):

    with open(

        filename,

        encoding="utf-8",

    ) as f:

        data = json.load(f)

    snapshot = GraphSnapshot()

    # ----------------------------------------------
    # Nodes
    # ----------------------------------------------

    for node in data["nodes"]:

        snapshot.nodes.append(

            SerializedNode(

                **node

            )

        )

    # ----------------------------------------------
    # Edges
    # ----------------------------------------------

    for edge in data["edges"]:

        snapshot.edges.append(

            SerializedEdge(

                **edge

            )

        )

    snapshot.statistics = SerializedGraphStatistics(
        **data["statistics"]
    )

    snapshot.signals = SerializedGraphSignals(
        **data["signals"]
    )

    return snapshot