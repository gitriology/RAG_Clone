from backend.optimization.adaptive_k import get_initial_k
from backend.optimization.confidence import calculate_confidence
from backend.retrieval.run_retrieval import retrieve


def adaptive_retrieve(query):

    # Decide initial K
    info = get_initial_k(query)

    current_k = info["initial_k"]

    max_k = 12

    while True:

        print(f"\nRetrieving Top-{current_k} documents...")

        docs = retrieve(query, top_k=current_k)

        # Placeholder confidence scores
        # We'll replace these with real retrieval scores later
        scores = [0.92 - (i * 0.05) for i in range(len(docs))]

        confidence = calculate_confidence(scores)

        print(confidence)

        if confidence["status"] != "Low":
            break

        current_k += 2

        if current_k > max_k:
            break

    return docs


if __name__ == "__main__":

    query = "Compare vaccination policies"

    results = adaptive_retrieve(query)

    print("\nRetrieved Documents")

    for d in results:
        print("-" * 50)
        print(d["text"][:200])