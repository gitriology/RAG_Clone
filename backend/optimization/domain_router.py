DOMAIN_KEYWORDS = {

    "healthcare": [
        "vaccine",
        "vaccination",
        "immunization",
        "covid",
        "virus",
        "patient",
        "doctor",
        "hospital",
        "disease",
        "health"
    ],

    "education": [
        "machine learning",
        "classification",
        "regression",
        "student",
        "teacher",
        "education",
        "pca",
        "knn",
        "decision tree",
        "learning"
    ],

    "administrative": [
        "employee",
        "policy",
        "leave",
        "salary",
        "hr",
        "recruitment",
        "performance",
        "organization"
    ]
}


def detect_domain(query):

    query = query.lower()

    scores = {}

    for domain, words in DOMAIN_KEYWORDS.items():

        score = 0

        for word in words:

            if word in query:
                score += 1

        scores[domain] = score

    best = max(scores, key=scores.get)

    if scores[best] == 0:
        return "all"

    return best