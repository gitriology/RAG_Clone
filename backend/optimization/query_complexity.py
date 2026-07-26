def analyze_query(query):
    """
    Estimate query complexity based on length and keywords.
    """

    query = query.lower()

    score = 0

    if len(query.split()) > 8:
        score += 1

    complex_words = [
        "compare",
        "difference",
        "advantages",
        "disadvantages",
        "relationship",
        "impact",
        "analysis",
        "policy",
        "versus",
        "explain"
    ]

    for word in complex_words:
        if word in query:
            score += 1

    if score <= 0:
        return "simple"

    elif score == 1:
        return "medium"

    return "complex"