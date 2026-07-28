from backend.ms_arc.state.retrieval_state import RetrievalState
from backend.ms_arc.retrieval.retrieve import retrieve
from backend.ms_arc.confidence.margin import compute_margin
from backend.evidence_graph.graph_builder import build_graph

state = RetrievalState(
    query="What is vaccination?"
)

state = retrieve(state)
state = compute_margin(state)

graph = build_graph(state)

report = graph.validation

print("=" * 60)
print("GRAPH VALIDATION")
print("=" * 60)

print("Valid            :", report.valid)
print("Passed Checks    :", report.passed_checks)
print("Failed Checks    :", report.failed_checks)
print("Warnings         :", report.warning_checks)
print("Validation Score :", round(report.validation_score, 3))

if report.errors:
    print("\nErrors:")
    for error in report.errors:
        print("-", error)

if report.warnings:
    print("\nWarnings:")
    for warning in report.warnings:
        print("-", warning)