"""
Optimization #6
Fusion Ablation Report

Reads fusion_benchmark_results.json and produces
a concise research-oriented summary.
"""

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

RESULT_FILE = (
    BASE_DIR /
    "fusion_benchmark_results.json"
)


def main():

    if not RESULT_FILE.exists():

        raise FileNotFoundError(
            "Run fusion_benchmark.py first."
        )

    with open(
        RESULT_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    minmax = data["results"]["minmax"]
    rrf = data["results"]["rrf"]

    comparison = data["comparison"]

    print()
    print("=" * 80)
    print("OPTIMIZATION #6 — FUSION ABLATION")
    print("=" * 80)

    print()

    print(
        f"Queries evaluated: "
        f"{minmax['queries_evaluated']}"
    )

    print(
        f"Top-K: "
        f"{minmax['top_k']}"
    )

    print()

    print(
        f"{'Metric':<20}"
        f"{'Min-Max':>12}"
        f"{'RRF':>12}"
        f"{'Delta':>12}"
    )

    print("-" * 60)

    metrics = [
        (
            "Recall@3",
            "recall_at_3",
        ),
        (
            "Precision@3",
            "precision_at_3",
        ),
        (
            "HitRate@3",
            "hit_rate_at_3",
        ),
        (
            "MRR@3",
            "mrr_at_3",
        ),
    ]

    for name, key in metrics:

        a = minmax[key]
        b = rrf[key]

        print(
            f"{name:<20}"
            f"{a:>12.4f}"
            f"{b:>12.4f}"
            f"{b - a:>12.4f}"
        )

    print()

    print(
        f"Min-Max average: "
        f"{comparison['minmax_average_score']:.4f}"
    )

    print(
        f"RRF average: "
        f"{comparison['rrf_average_score']:.4f}"
    )

    print()

    winner = comparison["winner"]

    print(
        f"Winner: "
        f"{winner.upper()}"
    )

    print()

    print("Research interpretation:")

    if winner == "rrf":

        print(
            "RRF outperformed Min-Max on the "
            "average retrieval evaluation score."
        )

    elif winner == "minmax":

        print(
            "Min-Max outperformed RRF on the "
            "average retrieval evaluation score."
        )

    else:

        print(
            "Both fusion strategies achieved "
            "the same average evaluation score."
        )

    print()

    print(
        "Important limitation:"
    )

    print(
        "Min-Max uses query-local normalization "
        "of dense and BM25 scores."
    )

    print(
        "Therefore, the comparison should be "
        "reported as an ablation rather than "
        "claiming Min-Max provides globally "
        "calibrated scores."
    )

    print()
    print("=" * 80)


if __name__ == "__main__":

    main()