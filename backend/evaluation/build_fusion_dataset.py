"""
Optimization #6
Step 7A - Fusion Evaluation Dataset Builder

Purpose
-------
Run evaluation queries through the existing retrieval pipeline
and display the retrieved document IDs/text so that relevant
documents can be manually identified.

IMPORTANT
---------
This script DOES NOT automatically decide ground truth.

Ground truth must be manually verified.

This prevents the benchmark from becoming circular:
retrieval results must not be treated automatically as truth.
"""

import json
from pathlib import Path

from backend.retrieval.run_retrieval import retrieve


# ==========================================================
# CONFIGURATION
# ==========================================================

OUTPUT_FILE = (
    Path(__file__).resolve().parent
    / "fusion_dataset.json"
)


EVALUATION_QUERIES = [

    # ------------------------------------------------------
    # Basic factual queries
    # ------------------------------------------------------

    "what is machine learning?",

    "what is WHO?",

    "what is ISRO?",

    "define machine learning",

    # ------------------------------------------------------
    # Conceptual queries
    # ------------------------------------------------------

    "difference between supervised and unsupervised learning",

    "what is supervised learning?",

    "what is unsupervised learning?",

    # ------------------------------------------------------
    # Domain-specific questions
    # ------------------------------------------------------

    "where did Chandrayaan-3 land?",

    "which organization developed Chandrayaan-3?",

    "what is the purpose of Chandrayaan-3?",

]


# ==========================================================
# RETRIEVAL DISPLAY
# ==========================================================

def collect_candidates():

    print()
    print("=" * 80)
    print("OPTIMIZATION #6 - EVALUATION DATASET BUILDER")
    print("=" * 80)

    dataset = []

    for index, query in enumerate(
        EVALUATION_QUERIES,
        start=1,
    ):

        print()
        print("=" * 80)

        print(
            f"QUERY {index}/{len(EVALUATION_QUERIES)}"
        )

        print(
            f"Query: {query}"
        )

        print("=" * 80)

        try:

            result = retrieve(query)

        except Exception as exc:

            print(
                "[ERROR] Retrieval failed:"
            )

            print(exc)

            continue

        documents = result.get(
            "documents",
            []
        )

        print()
        print(
            f"Retrieved documents: "
            f"{len(documents)}"
        )

        candidates = []

        for rank, document in enumerate(
            documents,
            start=1,
        ):

            # ----------------------------------------------
            # Document ID
            # ----------------------------------------------

            doc_id = document.get(
                "doc_id"
            )

            # Some downstream structures may not expose
            # doc_id. In that case we keep it as None.
            if doc_id is None:

                doc_id = document.get(
                    "id"
                )

            text = document.get(
                "text",
                ""
            )

            candidate = {

                "rank": rank,

                "doc_id": doc_id,

                "text": text,

            }

            candidates.append(
                candidate
            )

            print()
            print(
                f"--- Rank {rank} ---"
            )

            print(
                f"Document ID: {doc_id}"
            )

            print(
                "Text:"
            )

            print(
                text[:1000]
            )

        dataset.append({

            "query": query,

            "candidates": candidates,

            # ------------------------------------------------
            # IMPORTANT:
            #
            # This is intentionally empty.
            # Manually fill this after inspecting candidates.
            # ------------------------------------------------

            "relevant_doc_ids": [],

        })

    return dataset


# ==========================================================
# SAVE
# ==========================================================

def save_dataset(dataset):

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            dataset,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print("=" * 80)

    print(
        "Evaluation dataset created:"
    )

    print(
        OUTPUT_FILE
    )

    print("=" * 80)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    dataset = collect_candidates()

    save_dataset(
        dataset
    )