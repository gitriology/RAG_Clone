import numpy as np


def calculate_confidence(scores):
    """
    Calculates retrieval confidence from similarity scores.

    Parameters
    ----------
    scores : list
        Similarity scores returned by retrieval.

    Returns
    -------
    dict
    """

    if len(scores) == 0:
        return {
            "confidence": 0.0,
            "status": "Low"
        }

    confidence = float(np.mean(scores))

    if confidence >= 0.80:
        status = "High"

    elif confidence >= 0.60:
        status = "Medium"

    else:
        status = "Low"

    return {
        "confidence": round(confidence, 3),
        "status": status
    }


if __name__ == "__main__":

    retrieval_scores = [
        0.91,
        0.87,
        0.84,
        0.81,
        0.76
    ]

    result = calculate_confidence(retrieval_scores)

    print(result)