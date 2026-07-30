from backend.ms_arc.state.retrieval_state import (
    RetrievalState,
)

from backend.ms_arc.retrieval.retrieve import (
    retrieve,
)

from backend.ms_arc.confidence.margin import (
    compute_margin,
)

from backend.evidence_graph.graph_builder import (
    build_graph,
)

from backend.evidence_state.build_evidence_state import (
    build_evidence_state,
)

from backend.evidence_state.serialization.evidence_serializer import (
    save_evidence_state,
)

from backend.evidence_state.serialization.evidence_deserializer import (
    load_snapshot,
)
from backend.evidence_state.serialization.restore_evidence_state import (
    restore_evidence_state,
)

# ==========================================================
# Build Pipeline
# ==========================================================

retrieval = RetrievalState(
    query="What is vaccination?"
)

retrieval = retrieve(retrieval)

retrieval = compute_margin(retrieval)

graph = build_graph(retrieval)

evidence = build_evidence_state(
    retrieval,
    graph,
)


# ==========================================================
# Save
# ==========================================================

save_evidence_state(
    evidence,
    "evidence_snapshot.json",
)


print()

print("=" * 60)

print("Evidence State Saved")

print("=" * 60)

# ==========================================================
# Load
# ==========================================================

snapshot = load_snapshot(
    "evidence_snapshot.json",
)
restored = restore_evidence_state(
    snapshot
)
print()

print("=" * 60)

print("Evidence Snapshot Loaded")

print("=" * 60)

print()

print("Features :", len(snapshot.features))

print("Groups   :", len(snapshot.groups))

print("Vector Length :", len(snapshot.vector))

print("Evidence Score :", snapshot.signals.evidence_score)

print("Reasoning Ready :", snapshot.reasoning.readiness_score)
print("RESTORED")

print("="*60)

print()

print(restored.evidence_score)

print(restored.statistics.total_features)

print(restored.reasoning.summary.summary)

print(restored.reasoning.readiness.score)