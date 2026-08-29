from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from backend.retrieval.utils.load_embeddings import (
    load_embeddings,
)

from backend.retrieval.dense.embedder import (
    encode_query,
)

from backend.retrieval.index.faiss_index import (
    build_faiss,
    search_faiss,
    get_index_metadata,
)


# ==========================================================
# OPTIMIZATION #15
# ==========================================================

OPTIMIZATION_NAME = (
    "Optimization #15"
)

TOP_K = 3

SEARCH_RUNS = 30

WARMUP_RUNS = 5


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


RESULT_PATH = (
    PROJECT_ROOT
    / "optimization_15_results.json"
)


# ==========================================================
# EVALUATION DATASET
# ==========================================================
#
# These are the same five evaluation queries and relevant
# document ids used in the previous retrieval benchmark.
#
# The only variable in this benchmark is the FAISS index.
#
# ==========================================================

EVALUATION_QUERIES = [

    {
        "query":
            "what is machine learning?",

        "relevant_doc_ids":
            [6],
    },

    {
        "query":
            "what is WHO?",

        "relevant_doc_ids":
            [108],
    },

    {
        "query":
            "what is ISRO?",

        "relevant_doc_ids":
            [25],
    },

    {
        "query":
            "define machine learning",

        "relevant_doc_ids":
            [6],
    },

    {
        "query":
            "difference between supervised and unsupervised learning",

        "relevant_doc_ids":
            [114],
    },

]


# ==========================================================
# METRICS
# ==========================================================

def recall_at_k(
    retrieved_ids,
    relevant_ids,
):
    retrieved = set(
        retrieved_ids
    )

    relevant = set(
        relevant_ids
    )

    if not relevant:
        return 0.0

    return float(
        len(
            retrieved.intersection(
                relevant
            )
        )
        /
        len(relevant)
    )


def precision_at_k(
    retrieved_ids,
    relevant_ids,
):
    if not retrieved_ids:
        return 0.0

    relevant = set(
        relevant_ids
    )

    hits = sum(
        1
        for doc_id in retrieved_ids
        if doc_id in relevant
    )

    return float(
        hits
        /
        len(retrieved_ids)
    )


def hit_rate_at_k(
    retrieved_ids,
    relevant_ids,
):
    relevant = set(
        relevant_ids
    )

    return float(
        any(
            doc_id in relevant
            for doc_id in retrieved_ids
        )
    )


def mrr_at_k(
    retrieved_ids,
    relevant_ids,
):
    relevant = set(
        relevant_ids
    )

    for rank, doc_id in enumerate(
        retrieved_ids,
        start=1,
    ):

        if doc_id in relevant:

            return 1.0 / rank

    return 0.0


# ==========================================================
# BUILD INDEX
# ==========================================================

def build_index(
    embeddings,
    index_type,
):

    start = time.perf_counter()

    index = build_faiss(

        embeddings,

        index_type=index_type,

        hnsw_m=32,

        hnsw_ef_construction=40,

        hnsw_ef_search=32,

    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return index, elapsed


# ==========================================================
# SEARCH LATENCY
# ==========================================================

def measure_search_latency(
    index,
    query_embeddings,
):

    # ------------------------------------------------------
    # WARMUP
    # ------------------------------------------------------

    for _ in range(
        WARMUP_RUNS
    ):

        for query_embedding in query_embeddings:

            search_faiss(

                query_embedding,

                index,

                k=TOP_K,

            )

    # ------------------------------------------------------
    # MEASURE
    # ------------------------------------------------------

    start = time.perf_counter()

    total_searches = 0

    for _ in range(
        SEARCH_RUNS
    ):

        for query_embedding in query_embeddings:

            search_faiss(

                query_embedding,

                index,

                k=TOP_K,

            )

            total_searches += 1

    elapsed = (
        time.perf_counter()
        - start
    )

    if total_searches == 0:
        return 0.0

    return (
        elapsed
        /
        total_searches
        *
        1000.0
    )


# ==========================================================
# QUALITY EVALUATION
# ==========================================================

def evaluate_quality(
    index,
    query_embeddings,
):

    query_results = []

    recalls = []

    precisions = []

    hit_rates = []

    mrrs = []

    for evaluation, query_embedding in zip(
        EVALUATION_QUERIES,
        query_embeddings,
    ):

        scores, indices = search_faiss(

            query_embedding,

            index,

            k=TOP_K,

        )

        retrieved_ids = [

            int(doc_id)

            for doc_id in indices

            if int(doc_id) >= 0

        ]

        relevant_ids = [

            int(doc_id)

            for doc_id
            in evaluation[
                "relevant_doc_ids"
            ]

        ]

        recall = recall_at_k(

            retrieved_ids,

            relevant_ids,

        )

        precision = precision_at_k(

            retrieved_ids,

            relevant_ids,

        )

        hit_rate = hit_rate_at_k(

            retrieved_ids,

            relevant_ids,

        )

        mrr = mrr_at_k(

            retrieved_ids,

            relevant_ids,

        )

        recalls.append(
            recall
        )

        precisions.append(
            precision
        )

        hit_rates.append(
            hit_rate
        )

        mrrs.append(
            mrr
        )

        query_results.append({

            "query":
                evaluation["query"],

            "relevant_doc_ids":
                relevant_ids,

            "retrieved_doc_ids":
                retrieved_ids,

            "recall_at_3":
                recall,

            "precision_at_3":
                precision,

            "hit_rate_at_3":
                hit_rate,

            "mrr_at_3":
                mrr,

        })

    return {

        "recall_at_3":
            float(
                np.mean(recalls)
            ),

        "precision_at_3":
            float(
                np.mean(precisions)
            ),

        "hit_rate_at_3":
            float(
                np.mean(hit_rates)
            ),

        "mrr_at_3":
            float(
                np.mean(mrrs)
            ),

        "query_results":
            query_results,

    }


# ==========================================================
# BENCHMARK ONE INDEX
# ==========================================================

def benchmark_index(
    embeddings,
    query_embeddings,
    index_type,
):

    print()
    print(
        "--------------------------------------------------"
    )

    print(
        f"MODE: {index_type.upper()}"
    )

    print(
        "--------------------------------------------------"
    )

    # ------------------------------------------------------
    # BUILD
    # ------------------------------------------------------

    index, build_time = build_index(

        embeddings,

        index_type,

    )

    metadata = get_index_metadata(
        index
    )

    # ------------------------------------------------------
    # SEARCH LATENCY
    # ------------------------------------------------------

    search_latency_ms = (
        measure_search_latency(

            index,

            query_embeddings,

        )
    )

    # ------------------------------------------------------
    # QUALITY
    # ------------------------------------------------------

    quality = evaluate_quality(

        index,

        query_embeddings,

    )

    result = {

        "index_type":
            index_type,

        "index_metadata":
            metadata,

        "build_time_seconds":
            float(build_time),

        "average_search_latency_ms":
            float(
                search_latency_ms
            ),

        "top_k":
            TOP_K,

        "queries_total":
            len(EVALUATION_QUERIES),

        **quality,

    }

    print(
        f"Build time: "
        f"{build_time:.4f}s"
    )

    print(
        f"Average search latency: "
        f"{search_latency_ms:.4f} ms"
    )

    print(
        f"Recall@3: "
        f"{quality['recall_at_3']:.4f}"
    )

    print(
        f"Precision@3: "
        f"{quality['precision_at_3']:.4f}"
    )

    print(
        f"HitRate@3: "
        f"{quality['hit_rate_at_3']:.4f}"
    )

    print(
        f"MRR@3: "
        f"{quality['mrr_at_3']:.4f}"
    )

    return result


# ==========================================================
# WINNER
# ==========================================================

def select_winner(
    flat_result,
    hnsw_result,
):

    flat_quality = (

        flat_result["recall_at_3"],

        flat_result["precision_at_3"],

        flat_result["hit_rate_at_3"],

        flat_result["mrr_at_3"],

    )

    hnsw_quality = (

        hnsw_result["recall_at_3"],

        hnsw_result["precision_at_3"],

        hnsw_result["hit_rate_at_3"],

        hnsw_result["mrr_at_3"],

    )

    # ------------------------------------------------------
    # Quality has priority.
    #
    # We do NOT choose HNSW simply because it is faster.
    # An approximate index should not replace the exact
    # index if retrieval quality drops.
    # ------------------------------------------------------

    if hnsw_quality > flat_quality:

        return "hnsw"

    if flat_quality > hnsw_quality:

        return "flat"

    # ------------------------------------------------------
    # Equal quality -> choose faster search.
    # ------------------------------------------------------

    if (
        hnsw_result[
            "average_search_latency_ms"
        ]
        <
        flat_result[
            "average_search_latency_ms"
        ]
    ):

        return "hnsw"

    return "flat"


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()
    print(
        "=================================================="
    )

    print(
        "OPTIMIZATION #15"
    )

    print(
        "FAISS INDEX BENCHMARK"
    )

    print(
        "=================================================="
    )

    print(
        f"Project root: "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Top-K: {TOP_K}"
    )

    print(
        f"Queries: "
        f"{len(EVALUATION_QUERIES)}"
    )

    print(
        "Same embeddings: YES"
    )

    print(
        "Same queries: YES"
    )

    print(
        "Only index type changes: YES"
    )

    # ------------------------------------------------------
    # LOAD EMBEDDINGS
    # ------------------------------------------------------

    print()
    print(
        "Loading cached embeddings..."
    )

    embeddings = load_embeddings()

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    # ------------------------------------------------------
    # ENCODE QUERIES ONCE
    # ------------------------------------------------------

    print()
    print(
        "Encoding benchmark queries..."
    )

    query_texts = [

        item["query"]

        for item in EVALUATION_QUERIES

    ]

    query_embeddings = encode_query(
        query_texts
    )

    query_embeddings = np.asarray(
        query_embeddings,
        dtype=np.float32,
    )

    if query_embeddings.ndim == 1:

        query_embeddings = (
            query_embeddings.reshape(
                1,
                -1,
            )
        )

    print(
        f"Query embedding shape: "
        f"{query_embeddings.shape}"
    )

    # ------------------------------------------------------
    # FLAT
    # ------------------------------------------------------

    flat_result = benchmark_index(

        embeddings,

        query_embeddings,

        "flat",

    )

    # ------------------------------------------------------
    # HNSW
    # ------------------------------------------------------

    hnsw_result = benchmark_index(

        embeddings,

        query_embeddings,

        "hnsw",

    )

    # ------------------------------------------------------
    # WINNER
    # ------------------------------------------------------

    winner = select_winner(

        flat_result,

        hnsw_result,

    )

    # ------------------------------------------------------
    # DELTAS
    # ------------------------------------------------------

    search_speedup = 0.0

    flat_latency = (
        flat_result[
            "average_search_latency_ms"
        ]
    )

    hnsw_latency = (
        hnsw_result[
            "average_search_latency_ms"
        ]
    )

    if hnsw_latency > 0:

        search_speedup = (
            flat_latency
            /
            hnsw_latency
        )

    # ------------------------------------------------------
    # FINAL RESULT
    # ------------------------------------------------------

    results = {

        "optimization":
            OPTIMIZATION_NAME,

        "experiment":
            "IndexFlatIP vs HNSW",

        "evaluation_protocol": {

            "top_k":
                TOP_K,

            "same_dataset":
                True,

            "same_embeddings":
                True,

            "same_queries":
                True,

            "same_ground_truth":
                True,

            "only_variable":
                "faiss_index_type",

            "search_runs":
                SEARCH_RUNS,

            "warmup_runs":
                WARMUP_RUNS,

        },

        "dataset": {

            "documents":
                int(
                    embeddings.shape[0]
                ),

            "embedding_dimension":
                int(
                    embeddings.shape[1]
                ),

        },

        "results": {

            "flat":
                flat_result,

            "hnsw":
                hnsw_result,

        },

        "comparison": {

            "search_latency_speedup_hnsw_vs_flat":
                float(
                    search_speedup
                ),

            "recall_delta_hnsw_minus_flat":
                float(
                    hnsw_result[
                        "recall_at_3"
                    ]
                    -
                    flat_result[
                        "recall_at_3"
                    ]
                ),

            "precision_delta_hnsw_minus_flat":
                float(
                    hnsw_result[
                        "precision_at_3"
                    ]
                    -
                    flat_result[
                        "precision_at_3"
                    ]
                ),

            "hit_rate_delta_hnsw_minus_flat":
                float(
                    hnsw_result[
                        "hit_rate_at_3"
                    ]
                    -
                    flat_result[
                        "hit_rate_at_3"
                    ]
                ),

            "mrr_delta_hnsw_minus_flat":
                float(
                    hnsw_result[
                        "mrr_at_3"
                    ]
                    -
                    flat_result[
                        "mrr_at_3"
                    ]
                ),

            "winner":
                winner,

        },

    }

    # ------------------------------------------------------
    # PRINT SUMMARY
    # ------------------------------------------------------

    print()
    print(
        "=================================================="
    )

    print(
        "OPTIMIZATION #15 RESULTS"
    )

    print(
        "=================================================="
    )

    print(
        f"Flat search latency: "
        f"{flat_result['average_search_latency_ms']:.4f} ms"
    )

    print(
        f"HNSW search latency: "
        f"{hnsw_result['average_search_latency_ms']:.4f} ms"
    )

    print()

    print(
        f"Flat Recall@3: "
        f"{flat_result['recall_at_3']:.4f}"
    )

    print(
        f"HNSW Recall@3: "
        f"{hnsw_result['recall_at_3']:.4f}"
    )

    print()

    print(
        f"Flat MRR@3: "
        f"{flat_result['mrr_at_3']:.4f}"
    )

    print(
        f"HNSW MRR@3: "
        f"{hnsw_result['mrr_at_3']:.4f}"
    )

    print()

    print(
        "WINNER:",
        winner.upper(),
    )

    # ------------------------------------------------------
    # SAVE
    # ------------------------------------------------------

    with RESULT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(

            results,

            file,

            indent=4,

        )

    print()
    print(
        "Results saved to:"
    )

    print(
        RESULT_PATH
    )

    print()
    print(
        "Optimization #15 benchmark complete."
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()