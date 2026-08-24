"""
Optimization #6
Fusion Benchmark

Compares:

    Min-Max weighted fusion
    VS
    Reciprocal Rank Fusion (RRF)

Metrics:

    Recall@3
    Precision@3
    HitRate@3
    MRR@3

Both methods use:

    - identical evaluation queries
    - identical ground-truth documents
    - identical corpus
    - identical candidate-controller pipeline

Only the fusion method changes.
"""

import json
from pathlib import Path
from typing import Dict, List


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_FILE = (
    BASE_DIR /
    "fusion_dataset.json"
)

RESULT_FILE = (
    BASE_DIR /
    "fusion_benchmark_results.json"
)


# ==========================================================
# CONFIGURATION
# ==========================================================

TOP_K = 3

FUSION_METHODS = [
    "minmax",
    "rrf",
]


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset():

    if not DATASET_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: "
            f"{DATASET_FILE}"
        )

    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        dataset = json.load(file)

    if not isinstance(
        dataset,
        list,
    ):

        raise ValueError(
            "fusion_dataset.json "
            "must contain a JSON list."
        )

    return dataset


# ==========================================================
# VALIDATE DATASET
# ==========================================================

def validate_dataset(
    dataset,
):

    valid_queries = []

    for item in dataset:

        query = item.get(
            "query"
        )

        relevant_ids = item.get(
            "relevant_doc_ids",
            [],
        )

        if not query:

            print(
                "[WARNING] "
                "Skipping item with no query."
            )

            continue

        if not relevant_ids:

            print(
                "[WARNING] "
                f"No ground truth for: "
                f"{query}"
            )

            continue

        valid_queries.append({

            "query":
                query,

            "relevant_doc_ids":
                [
                    int(x)
                    for x in relevant_ids
                ],

        })

    if not valid_queries:

        raise ValueError(
            "No labelled evaluation "
            "queries found."
        )

    return valid_queries


# ==========================================================
# RETRIEVAL
# ==========================================================

def retrieve_for_method(
    query: str,
    fusion_method: str,
):
    """
    Run the COMPLETE existing retrieval
    pipeline.

    IMPORTANT:
        fusion_method is explicitly propagated
        into the retrieval pipeline.
    """

    from backend.retrieval.run_retrieval import (
        retrieve,
    )

    result = retrieve(
        query=query,
        fusion_method=fusion_method,
    )

    if not isinstance(
        result,
        dict,
    ):

        raise TypeError(
            "retrieve() must return "
            "a dictionary."
        )

    documents = result.get(
        "documents",
        [],
    )

    if not isinstance(
        documents,
        list,
    ):

        raise TypeError(
            "retrieve()['documents'] "
            "must be a list."
        )

    return documents


# ==========================================================
# DOC ID EXTRACTION
# ==========================================================

def get_doc_id(
    document,
):

    if not isinstance(
        document,
        dict,
    ):

        return None

    doc_id = document.get(
        "doc_id"
    )

    if doc_id is None:

        doc_id = document.get(
            "id"
        )

    if doc_id is None:

        return None

    try:

        return int(doc_id)

    except (
        TypeError,
        ValueError,
    ):

        return None


# ==========================================================
# RECALL@K
# ==========================================================

def recall_at_k(
    retrieved_ids: List[int],
    relevant_ids: List[int],
    k: int,
):

    relevant = set(
        relevant_ids
    )

    if not relevant:

        return 0.0

    retrieved = set(
        retrieved_ids[:k]
    )

    return (
        len(
            retrieved & relevant
        )
        /
        len(relevant)
    )


# ==========================================================
# PRECISION@K
# ==========================================================

def precision_at_k(
    retrieved_ids: List[int],
    relevant_ids: List[int],
    k: int,
):

    retrieved = retrieved_ids[:k]

    if not retrieved:

        return 0.0

    relevant = set(
        relevant_ids
    )

    hits = sum(
        1
        for doc_id in retrieved
        if doc_id in relevant
    )

    return (
        hits
        /
        len(retrieved)
    )


# ==========================================================
# HIT RATE@K
# ==========================================================

def hit_rate_at_k(
    retrieved_ids: List[int],
    relevant_ids: List[int],
    k: int,
):

    retrieved = set(
        retrieved_ids[:k]
    )

    relevant = set(
        relevant_ids
    )

    if not relevant:

        return 0.0

    return (
        1.0
        if retrieved & relevant
        else 0.0
    )


# ==========================================================
# MRR@K
# ==========================================================

def mrr_at_k(
    retrieved_ids: List[int],
    relevant_ids: List[int],
    k: int,
):

    relevant = set(
        relevant_ids
    )

    for rank, doc_id in enumerate(
        retrieved_ids[:k],
        start=1,
    ):

        if doc_id in relevant:

            return 1.0 / rank

    return 0.0


# ==========================================================
# EVALUATE METHOD
# ==========================================================

def evaluate_method(
    dataset,
    fusion_method,
):

    print()
    print("=" * 80)

    print(
        f"Evaluating fusion method: "
        f"{fusion_method.upper()}"
    )

    print("=" * 80)

    recall_values = []
    precision_values = []
    hit_values = []
    mrr_values = []

    query_results = []

    successful_queries = 0
    failed_queries = 0

    for index, item in enumerate(
        dataset,
        start=1,
    ):

        query = item["query"]

        relevant_ids = [
            int(x)
            for x in item[
                "relevant_doc_ids"
            ]
        ]

        print()
        print(
            f"[{index}/{len(dataset)}] "
            f"{query}"
        )

        try:

            documents = retrieve_for_method(
                query,
                fusion_method,
            )

        except Exception as exc:

            failed_queries += 1

            print(
                "[ERROR] Retrieval failed:"
            )

            print(exc)

            continue

        retrieved_ids = []

        for document in documents:

            doc_id = get_doc_id(
                document
            )

            if doc_id is None:

                print(
                    "[WARNING] "
                    "Document missing "
                    "valid doc_id."
                )

                continue

            retrieved_ids.append(
                doc_id
            )

        if not retrieved_ids:

            failed_queries += 1

            print(
                "[WARNING] No valid "
                "retrieved document IDs."
            )

            continue

        successful_queries += 1

        recall = recall_at_k(
            retrieved_ids,
            relevant_ids,
            TOP_K,
        )

        precision = precision_at_k(
            retrieved_ids,
            relevant_ids,
            TOP_K,
        )

        hit = hit_rate_at_k(
            retrieved_ids,
            relevant_ids,
            TOP_K,
        )

        mrr = mrr_at_k(
            retrieved_ids,
            relevant_ids,
            TOP_K,
        )

        recall_values.append(
            recall
        )

        precision_values.append(
            precision
        )

        hit_values.append(
            hit
        )

        mrr_values.append(
            mrr
        )

        query_results.append({

            "query":
                query,

            "relevant_doc_ids":
                relevant_ids,

            "retrieved_doc_ids":
                retrieved_ids,

            "fusion_method":
                fusion_method,

            "recall_at_3":
                recall,

            "precision_at_3":
                precision,

            "hit_rate_at_3":
                hit,

            "mrr_at_3":
                mrr,

        })

        print(
            f"Retrieved: "
            f"{retrieved_ids[:TOP_K]}"
        )

        print(
            f"Recall@3: "
            f"{recall:.4f}"
        )

        print(
            f"Precision@3: "
            f"{precision:.4f}"
        )

        print(
            f"HitRate@3: "
            f"{hit:.4f}"
        )

        print(
            f"MRR@3: "
            f"{mrr:.4f}"
        )

    if successful_queries == 0:

        raise RuntimeError(
            f"No queries successfully "
            f"evaluated for {fusion_method}."
        )

    return {

        "fusion_method":
            fusion_method,

        "queries_total":
            len(dataset),

        "queries_evaluated":
            successful_queries,

        "queries_failed":
            failed_queries,

        "top_k":
            TOP_K,

        "recall_at_3":
            sum(recall_values)
            / len(recall_values),

        "precision_at_3":
            sum(precision_values)
            / len(precision_values),

        "hit_rate_at_3":
            sum(hit_values)
            / len(hit_values),

        "mrr_at_3":
            sum(mrr_values)
            / len(mrr_values),

        "query_results":
            query_results,

    }


# ==========================================================
# COMPARE
# ==========================================================

def compare_results(
    results: Dict,
):

    minmax = results["minmax"]

    rrf = results["rrf"]

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

    print()
    print("=" * 80)
    print("OPTIMIZATION #6")
    print("MIN-MAX VS RRF")
    print("=" * 80)

    print()

    print(
        f"{'Metric':<20}"
        f"{'Min-Max':>12}"
        f"{'RRF':>12}"
        f"{'Delta':>12}"
    )

    print("-" * 60)

    metric_comparison = {}

    for name, key in metrics:

        minmax_value = minmax[key]
        rrf_value = rrf[key]

        delta = (
            rrf_value
            - minmax_value
        )

        metric_comparison[key] = {

            "minmax":
                minmax_value,

            "rrf":
                rrf_value,

            "delta_rrf_minus_minmax":
                delta,

        }

        print(
            f"{name:<20}"
            f"{minmax_value:>12.4f}"
            f"{rrf_value:>12.4f}"
            f"{delta:>12.4f}"
        )

    # ======================================================
    # OVERALL SCORE
    # ======================================================

    score_keys = [
        key
        for _, key in metrics
    ]

    minmax_score = sum(
        minmax[key]
        for key in score_keys
    ) / len(score_keys)

    rrf_score = sum(
        rrf[key]
        for key in score_keys
    ) / len(score_keys)

    if rrf_score > minmax_score:

        winner = "rrf"

    elif minmax_score > rrf_score:

        winner = "minmax"

    else:

        winner = "tie"

    print()
    print(
        f"Average Min-Max score: "
        f"{minmax_score:.4f}"
    )

    print(
        f"Average RRF score: "
        f"{rrf_score:.4f}"
    )

    print()

    print(
        f"Preliminary winner: "
        f"{winner.upper()}"
    )

    return {

        "metrics":
            metric_comparison,

        "minmax_average_score":
            minmax_score,

        "rrf_average_score":
            rrf_score,

        "winner":
            winner,

    }


# ==========================================================
# SAVE
# ==========================================================

def save_results(
    results,
    comparison,
):

    output = {

        "optimization":
            "Optimization #6",

        "experiment":
            "Min-Max vs RRF",

        "evaluation_protocol": {

            "top_k":
                TOP_K,

            "same_dataset":
                True,

            "same_ground_truth":
                True,

            "same_retrieval_pipeline":
                True,

            "only_variable":
                "fusion_method",

        },

        "results":
            results,

        "comparison":
            comparison,

    }

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print(
        "Benchmark results saved to:"
    )

    print(
        RESULT_FILE
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    dataset = load_dataset()

    dataset = validate_dataset(
        dataset
    )

    print()
    print(
        f"Loaded "
        f"{len(dataset)} labelled queries."
    )

    results = {}

    for fusion_method in FUSION_METHODS:

        results[
            fusion_method
        ] = evaluate_method(
            dataset,
            fusion_method,
        )

    comparison = compare_results(
        results
    )

    save_results(
        results,
        comparison,
    )

    print()
    print("=" * 80)
    print("Benchmark completed.")
    print("=" * 80)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()