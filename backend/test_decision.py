from backend.ms_arc.state.retrieval_state import RetrievalState

from backend.ms_arc.retrieval.retrieve import retrieve

from backend.ms_arc.agreement.agreement import compute_agreement

from backend.ms_arc.confidence.margin import compute_margin

from backend.ms_arc.stability.stability import compute_stability

from backend.ms_arc.decision.decision_engine import compute_decision


state = RetrievalState(

    query="What is vaccination?"

)

state.recommended_topk = 5

state = retrieve(state)

state = compute_agreement(state)

state = compute_margin(state)

state = compute_stability(state)

state = compute_decision(state)


print()

print("=" * 60)

print("Decision Engine")

print("=" * 60)

print("Confidence :", round(state.signals.decision.confidence, 4))

print("Decision   :", state.signals.decision.decision)

print("Reason     :", state.signals.decision.reason)

print()

print("Agreement  :", round(state.signals.decision.agreement, 4))

print("Margin     :", round(state.signals.decision.margin, 4))

print("Stability  :", round(state.signals.decision.stability, 4))