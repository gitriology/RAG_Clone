from backend.ms_arc.state.retrieval_state import RetrievalState
from backend.ms_arc.retrieval.retrieve import retrieve
from backend.ms_arc.stability.stability import compute_stability


state = RetrievalState(
    query="What is vaccination?"
)

state.recommended_topk = 5

state = retrieve(state)

state = compute_stability(state)

print("=" * 60)
print("Retrieval Stability")
print("=" * 60)

print(f"Stability Score : {state.signals.stability.score:.4f}")

print()

print(state.signals.stability.details)