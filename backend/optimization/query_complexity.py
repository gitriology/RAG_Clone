import re

# Keywords that usually indicate a complex query
COMPLEX_WORDS = [
    "compare",
    "difference",
    "advantages",
    "disadvantages",
    "explain",
    "analyze",
    "evaluation",
    "why",
    "how",
    "policy",
    "guidelines",
    "framework",
    "multiple"
]


def analyze_query(query):
    """
    Analyze the complexity of a user query.

    Returns:
        complexity : simple / medium / complex
        score : numerical score
    """

    query = query.lower()

    # Count number of words
    word_count = len(query.split())

    # Count complex keywords
    keyword_count = sum(
        1 for word in COMPLEX_WORDS
        if re.search(rf"\b{word}\b", query)
    )

    # Simple scoring formula
    score = (word_count * 0.05) + (keyword_count * 0.3)

    # Decide complexity
    if score < 0.5:
        complexity = "simple"

    elif score < 1.2:
        complexity = "medium"

    else:
        complexity = "complex"

    return {
        "complexity": complexity,
        "score": round(score, 2)
    }


if __name__ == "__main__":

    queries = [
        "What is vaccination?",
        "Explain vaccination schedule",
        "Compare WHO vaccination policy with India's immunization policy"
    ]

    for q in queries:

        result = analyze_query(q)

        print("\nQuery :", q)
        print("Complexity :", result["complexity"])
        print("Score :", result["score"])