from ms_arc.run_msarc import run_msarc

query = """
Explain the difference between FAISS retrieval and BM25 retrieval
in Retrieval Augmented Generation and compare their impact on latency.
"""

state = run_msarc(query)

print("\n========== MS-ARC ==========")

print("Query Type :", state.query_type)

print("Complexity :", state.query_complexity)

print("Top-K :", state.recommended_topk)

print()

print(state.debug)