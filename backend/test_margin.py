from backend.ms_arc.state.retrieval_state import RetrievalState

from backend.ms_arc.retrieval.retrieve import retrieve

from backend.ms_arc.agreement.agreement import compute_agreement

from backend.ms_arc.confidence.margin import compute_margin


state = RetrievalState(
    query="What is vaccination?"
)

state.recommended_topk = 5

state = retrieve(state)

state = compute_agreement(state)

state = compute_margin(state)

print()

print("=" * 60)
print("Margin Analysis")
print("=" * 60)

print("Top Score")
print(state.signals.margin.details["top_score"])

print()

print("Second Score")
print(state.signals.margin.details["second_score"])

print()

print("Margin")
print(state.signals.margin.raw_margin)

print()

print("Normalized Margin")
print(state.signals.margin.normalized_margin)

print()

print("Top Documents")

for doc in state.reranked_results:

    print()

    print(doc.doc_id)

    print(round(doc.rerank_score, 3))

    print(doc.text[:120])