def compute_evidence_state(retrieved_docs):
    """
    Computes an Evidence State score based on
    retrieval quality.
    """

    num_docs = len(retrieved_docs)

    domains = set()

    total_length = 0

    for doc in retrieved_docs:

        domains.add(doc["domain"])

        total_length += len(doc["text"].split())

    avg_length = total_length / max(num_docs, 1)

    diversity_score = len(domains) / max(num_docs, 1)

    # Normalize coverage
    coverage_score = min(num_docs / 10, 1)

    # Normalize text richness
    richness_score = min(avg_length / 250, 1)

    evidence_score = (
        0.40 * coverage_score +
        0.30 * richness_score +
        0.30 * diversity_score
    )

    return {
        "coverage": round(coverage_score, 3),
        "richness": round(richness_score, 3),
        "diversity": round(diversity_score, 3),
        "evidence_score": round(evidence_score, 3)
    }