"""CLI for Optimization #23 feature-weight tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.evidence_state.weighting.feature_weights import CANONICAL_FEATURE_WEIGHTS
from backend.evidence_state.weighting.feature_weight_tuner import (
    accuracy,
    compare_weight_sets,
    load_validation_dataset,
    save_weights,
    tune_feature_weights,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune EvidenceState feature weights from labelled validation data."
    )
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parent / "feature_weight_validation.example.json"),
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "tuned_feature_weights.json"),
    )
    args = parser.parse_args()

    dataset = load_validation_dataset(args.dataset)

    feature_names = sorted({
        str(name)
        for row in dataset
        for name in row["features"].keys()
    })
    baseline_subset = {
        name: CANONICAL_FEATURE_WEIGHTS.get(name, 1.0)
        for name in feature_names
    }
    equal = {name: 1.0 for name in feature_names}
    tuned = tune_feature_weights(
        dataset,
        baseline_subset,
        feature_names=feature_names,
    )

    comparison = compare_weight_sets(
        dataset,
        equal_weights=equal,
        manual_weights=baseline_subset,
        tuned_weights=tuned,
    )

    save_weights(
        tuned,
        args.output,
        metadata={
            "validation_records": len(dataset),
            "equal_accuracy": comparison["equal"],
            "manual_accuracy": comparison["manual"],
            "tuned_accuracy": comparison["tuned"],
        },
    )

    print("=" * 72)
    print("OPTIMIZATION #23 - FEATURE-WEIGHT TUNING")
    print("=" * 72)
    print(f"Validation records: {len(dataset)}")
    print(f"Equal weights accuracy:   {comparison['equal']:.4f}")
    print(f"Manual weights accuracy:  {comparison['manual']:.4f}")
    print(f"Tuned weights accuracy:   {comparison['tuned']:.4f}")
    print()
    print(f"Tuned weights saved to: {args.output}")
    print()
    for name, value in tuned.items():
        print(f"{name:<28} {value:.6f}")


if __name__ == "__main__":
    main()
