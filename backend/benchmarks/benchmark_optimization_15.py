from __future__ import annotations

import json
import time
from collections import defaultdict
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

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "all_domains.json"
)

FUSION_DATASET_PATH = (
    PROJECT_ROOT
    / "backend"
    / "evaluation"
    / "fusion_dataset.json"
)


# ==========================================================
# ID HELPERS
# ==========================================================

def normalize_id(value):

    if value is None:
        return None

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):

        return str(value)


def find_text_rows(
    data,
    text,
):
    """
    Find exact dataset rows for a candidate document.
    """

    return [

        row_index

        for row_index, document
        in enumerate(data)

        if document.get("text") == text

    ]


def resolve_ground_truth(
    data,
    evaluation,
):
    """
    Convert dataset-level ground truth into FAISS row IDs.

    IMPORTANT:

        evaluation["relevant_doc_ids"]

    contains dataset IDs.

    FAISS returns row positions.

    Since dataset IDs are duplicated, exact candidate text
    is used to identify the correct row.
    """

    relevant_ids = [

        normalize_id(value)

        for value
        in evaluation.get(
            "relevant_doc_ids",
            [],
        )

    ]

    candidates = evaluation.get(
        "candidates",
        [],
    )

    resolved_rows = []

    for relevant_id in relevant_ids:

        candidate_matches = [

            candidate

            for candidate in candidates

            if normalize_id(
                candidate.get("doc_id")
            ) == relevant_id

        ]

        candidate_rows = []

        for candidate in candidate_matches:

            candidate_text = candidate.get(
                "text",
                "",
            )

            if not candidate_text:
                continue

            candidate_rows.extend(
                find_text_rows(
                    data,
                    candidate_text,
                )
            )

        candidate_rows = sorted(
            set(candidate_rows)
        )

        if len(candidate_rows) != 1:

            raise ValueError(

                "\n"
                "GROUND-TRUTH ID RESOLUTION FAILED\n"
                f"Query: {evaluation['query']}\n"
                f"Dataset ID: {relevant_id}\n"
                f"Possible matching FAISS rows: "
                f"{candidate_rows}\n\n"
                "The dataset ID is duplicated and the "
                "ground truth could not be uniquely resolved. "
                "The benchmark will stop rather than guess."
            )

        resolved_rows.append(
            candidate_rows[0]
        )

    return resolved_rows


# ==========================================================
# LOAD EVALUATION DATA
# ==========================================================

def load_evaluation_data():

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    with FUSION_DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        fusion_dataset = json.load(
            file
        )

    evaluations = []

    for item in fusion_dataset:

        if not item.get(
            "relevant_doc_ids"
        ):

            continue

        faiss_rows = resolve_ground_truth(
            data,
            item,
        )

        evaluations.append({

            "query":
                item["query"],

            "relevant_dataset_ids":
                [
                    normalize_id(value)
                    for value
                    in item[
                        "relevant_doc_ids"
                    ]
                ],

            "relevant_faiss_rows":
                faiss_rows,

        })

    if not evaluations:

        raise ValueError(
            "No evaluation queries with ground truth "
            "were found."
        )

    return (
        data,
        evaluations,
    )


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

        for doc_id
        in retrieved_ids

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

        hnsw_m=64,

        hnsw_ef_construction=40,

        hnsw_ef_search=32,

    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        index,
        elapsed,
    )


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
    evaluations,
):

    query_results = []

    recalls = []

    precisions = []

    hit_rates = []

    mrrs = []

    for evaluation, query_embedding in zip(
        evaluations,
        query_embeddings,
    ):

        scores, indices = search_faiss(

            query_embedding,

            index,

            k=TOP_K,

        )

        retrieved_rows = [

            int(row)

            for row in indices

            if int(row) >= 0

        ]

        relevant_rows = [

            int(row)

            for row
            in evaluation[
                "relevant_faiss_rows"
            ]

        ]

        recall = recall_at_k(

            retrieved_rows,

            relevant_rows,

        )

        precision = precision_at_k(

            retrieved_rows,

            relevant_rows,

        )

        hit_rate = hit_rate_at_k(

            retrieved_rows,

            relevant_rows,

        )

        mrr = mrr_at_k(

            retrieved_rows,

            relevant_rows,

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

            "relevant_dataset_ids":
                evaluation[
                    "relevant_dataset_ids"
                ],

            "relevant_faiss_row_ids":
                relevant_rows,

            "retrieved_faiss_row_ids":
                retrieved_rows,

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
    evaluations,
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

    index, build_time = build_index(

        embeddings,

        index_type,

    )

    metadata = get_index_metadata(
        index
    )

    search_latency_ms = (
        measure_search_latency(

            index,

            query_embeddings,

        )
    )

    quality = evaluate_quality(

        index,

        query_embeddings,

        evaluations,

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
            len(evaluations),

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

        flat_result[
            "recall_at_3"
        ],

        flat_result[
            "precision_at_3"
        ],

        flat_result[
            "hit_rate_at_3"
        ],

        flat_result[
            "mrr_at_3"
        ],

    )

    hnsw_quality = (

        hnsw_result[
            "recall_at_3"
        ],

        hnsw_result[
            "precision_at_3"
        ],

        hnsw_result[
            "hit_rate_at_3"
        ],

        hnsw_result[
            "mrr_at_3"
        ],

    )

    # ------------------------------------------------------
    # Quality has priority.
    # ------------------------------------------------------

    if hnsw_quality > flat_quality:

        return "hnsw"

    if flat_quality > hnsw_quality:

        return "flat"

    # ------------------------------------------------------
    # Equal quality -> faster index wins.
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
        f"Search runs: {SEARCH_RUNS}"
    )

    print(
        f"Warmup runs: {WARMUP_RUNS}"
    )

    # ======================================================
    # LOAD DATASET + GROUND TRUTH
    # ======================================================

    print()
    print(
        "Resolving evaluation ground truth..."
    )

    (
        data,
        evaluations,
    ) = load_evaluation_data()

    print(
        f"Dataset documents: "
        f"{len(data)}"
    )

    print(
        f"Evaluation queries: "
        f"{len(evaluations)}"
    )

    print()
    print(
        "Ground-truth FAISS row mapping:"
    )

    for evaluation in evaluations:

        print(
            f"  {evaluation['query']}"
        )

        print(
            f"    dataset IDs: "
            f"{evaluation['relevant_dataset_ids']}"
        )

        print(
            f"    FAISS rows: "
            f"{evaluation['relevant_faiss_rows']}"
        )

    # ======================================================
    # LOAD EMBEDDINGS
    # ======================================================

    print()
    print(
        "Loading cached embeddings..."
    )

    embeddings = load_embeddings()

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    if len(data) != embeddings.shape[0]:

        raise ValueError(
            "Dataset/embedding row mismatch."
        )

    # ======================================================
    # ENCODE QUERIES ONCE
    # ======================================================

    print()

    print(
        "Encoding benchmark queries..."
    )

    query_texts = [
        evaluation["query"]
        for evaluation in evaluations
    ]

    # ------------------------------------------------------
    # IMPORTANT:
    #
    # encode_query() in the current project is designed
    # around a single query. Passing the complete list to it
    # produces a single embedding instead of one embedding
    # per query.
    #
    # Therefore encode each benchmark query individually
    # here and stack the results.
    #
    # This keeps the benchmark deterministic and ensures:
    #
    #     len(query_embeddings)
    #     ==
    #     len(evaluations)
    # ------------------------------------------------------

    query_embeddings_list = []

    for query_index, query_text in enumerate(
        query_texts
    ):

        print(
            f"[Benchmark] Encoding query "
            f"{query_index + 1}/{len(query_texts)}: "
            f"{query_text}"
        )

        embedding = encode_query(
            query_text
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Normalize shape.
        #
        # Accept:
        #
        #     (768,)
        # or:
        #     (1, 768)
        # --------------------------------------------------

        if embedding.ndim == 2:

            if embedding.shape[0] != 1:

                raise ValueError(
                    "encode_query() returned an unexpected "
                    f"batch shape for query {query_index}: "
                    f"{embedding.shape}"
                )

            embedding = embedding[0]

        elif embedding.ndim != 1:

            raise ValueError(
                "encode_query() returned an unexpected "
                f"shape for query {query_index}: "
                f"{embedding.shape}"
            )

        query_embeddings_list.append(
            embedding
        )

    # ------------------------------------------------------
    # Stack into:
    #
    #     (number_of_queries, embedding_dimension)
    #
    # Expected here:
    #
    #     (5, 768)
    # ------------------------------------------------------

    if not query_embeddings_list:

        raise ValueError(
            "No query embeddings were generated."
        )

    query_embeddings = np.stack(
        query_embeddings_list,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    print(
        f"Query embedding shape: "
        f"{query_embeddings.shape}"
    )

    # ------------------------------------------------------
    # FINAL ALIGNMENT CHECK
    # ------------------------------------------------------

    if len(query_embeddings) != len(
        evaluations
    ):

        raise ValueError(
            "Number of query embeddings does not "
            "match evaluation queries: "
            f"{len(query_embeddings)} embeddings "
            f"for {len(evaluations)} evaluations."
        )

    print(
        "[Benchmark] Query/evaluation alignment: PASS"
    )

    print(
        f"[Benchmark] Query embeddings: "
        f"{len(query_embeddings)}"
    )

    print(
        f"[Benchmark] Evaluation queries: "
        f"{len(evaluations)}"
    )
    # ======================================================
    # FLAT
    # ======================================================

    flat_result = benchmark_index(

        embeddings,

        query_embeddings,

        evaluations,

        "flat",

    )

    # ======================================================
    # HNSW
    # ======================================================

    hnsw_result = benchmark_index(

        embeddings,

        query_embeddings,

        evaluations,

        "hnsw",

    )

    # ======================================================
    # WINNER
    # ======================================================

    winner = select_winner(

        flat_result,

        hnsw_result,

    )

    # ======================================================
    # SPEEDUP
    # ======================================================

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

    search_speedup = 0.0

    if hnsw_latency > 0:

        search_speedup = (
            flat_latency
            /
            hnsw_latency
        )

    # ======================================================
    # RESULTS
    # ======================================================

    results = {

        "optimization":
            OPTIMIZATION_NAME,

        "experiment":
            "IndexFlatIP vs HNSW",

        "id_alignment":

            {

                "evaluation_ids_are":
                    "dataset_id",

                "faiss_results_are":
                    "faiss_row_id",

                "duplicate_dataset_ids":
                    True,

                "ground_truth_resolved_using":
                    "exact_candidate_text",

            },

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

    # ======================================================
    # SUMMARY
    # ======================================================

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
        f"Flat Precision@3: "
        f"{flat_result['precision_at_3']:.4f}"
    )

    print(
        f"HNSW Precision@3: "
        f"{hnsw_result['precision_at_3']:.4f}"
    )

    print()

    print(
        f"Flat HitRate@3: "
        f"{flat_result['hit_rate_at_3']:.4f}"
    )

    print(
        f"HNSW HitRate@3: "
        f"{hnsw_result['hit_rate_at_3']:.4f}"
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
        f"HNSW search speedup: "
        f"{search_speedup:.2f}x"
    )

    print()

    print(
        "WINNER:",
        winner.upper(),
    )

    # ======================================================
    # SAVE
    # ======================================================

    with RESULT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
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