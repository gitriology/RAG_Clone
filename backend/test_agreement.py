from backend.ms_arc.state.retrieval_state import RetrievalState
from backend.ms_arc.retrieval.retrieve import retrieve
from backend.ms_arc.agreement.agreement import compute_agreement


state = RetrievalState(

    query="What is WHO?"

)

state.recommended_topk = 5

state = retrieve(state)

state = compute_agreement(state)

print("=" * 60)
print("Weighted Agreement")
print("=" * 60)

print(state.signals.agreement.score)

print("Intersection :", state.signals.agreement.intersection)

print("Union :", state.signals.agreement.union)

print()

print("Intersection")

dense = {
    d.doc_id
    for d in state.dense_results
}

sparse = {
    d.doc_id
    for d in state.sparse_results
}

print(dense & sparse)

print()

print("Dense:", len(dense))

print("Sparse:", len(sparse))