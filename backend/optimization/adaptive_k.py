from backend.optimization.query_complexity import analyze_query


def get_initial_k(query):
    """
    Select initial retrieval size (K)
    based on query complexity.
    """

    result = analyze_query(query)

    complexity = result["complexity"]

    if complexity == "simple":
        k = 3

    elif complexity == "medium":
        k = 5

    else:
        k = 8

    return {
        "complexity": complexity,
        "score": result["score"],
        "initial_k": k
    }


if __name__ == "__main__":

    queries = [
        "What is vaccination?",
        "Explain vaccination schedule",
        "Compare WHO vaccination policy with India's immunization policy",
        "Compare WHO, CDC and NHA vaccination policies and explain the differences in immunization schedules"
    ]

    for q in queries:

        result = get_initial_k(q)

        print("\nQuery :", q)
        print("Complexity :", result["complexity"])
        print("Score :", result["score"])
        print("Initial K :", result["initial_k"])