from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import numpy as np

from backend.retrieval.utils.load_data import load_data


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "all_domains.json"
)

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "backend"
    / "retrieval"
    / "cache"
    / "embeddings.npy"
)

FUSION_DATASET_PATH = (
    PROJECT_ROOT
    / "backend"
    / "evaluation"
    / "fusion_dataset.json"
)


# ==========================================================
# HELPERS
# ==========================================================

def normalize_id(value):
    """
    Normalize IDs for comparison.

    Dataset IDs may be stored as integers while evaluation
    files may contain strings.
    """

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def build_id_to_rows(data):
    """
    Build:

        dataset id -> list of FAISS row indices

    Important:
        dataset IDs are NOT assumed to be unique.
    """

    mapping = defaultdict(list)

    for row_index, document in enumerate(data):

        document_id = normalize_id(
            document.get("id")
        )

        mapping[document_id].append(
            row_index
        )

    return dict(mapping)


def find_text_rows(data, text):
    """
    Find dataset rows containing the exact evaluation text.
    """

    matches = []

    for row_index, document in enumerate(data):

        if document.get("text") == text:

            matches.append(
                row_index
            )

    return matches


def resolve_ground_truth(
    data,
    fusion_item,
):
    """
    Resolve manually verified ground truth into FAISS row IDs.

    The fusion dataset stores the original dataset `doc_id`
    plus candidate text.

    Because dataset IDs are duplicated across source documents,
    the text is used to disambiguate the correct row.
    """

    relevant_ids = [
        normalize_id(value)
        for value in fusion_item.get(
            "relevant_doc_ids",
            [],
        )
    ]

    candidates = fusion_item.get(
        "candidates",
        [],
    )

    id_to_rows = build_id_to_rows(
        data
    )

    resolved_rows = []

    for relevant_id in relevant_ids:

        matching_candidates = [

            candidate

            for candidate in candidates

            if normalize_id(
                candidate.get("doc_id")
            ) == relevant_id

        ]

        # --------------------------------------------------
        # First choice:
        # exact candidate text match
        # --------------------------------------------------

        candidate_rows = []

        for candidate in matching_candidates:

            candidate_text = candidate.get(
                "text",
                "",
            )

            if not candidate_text:
                continue

            rows = find_text_rows(
                data,
                candidate_text,
            )

            candidate_rows.extend(
                rows
            )

        candidate_rows = sorted(
            set(candidate_rows)
        )

        if len(candidate_rows) == 1:

            resolved_rows.append(
                candidate_rows[0]
            )

            continue

        # --------------------------------------------------
        # If exact text did not uniquely resolve the ID,
        # report the ambiguity instead of guessing.
        # --------------------------------------------------

        possible_rows = id_to_rows.get(
            relevant_id,
            [],
        )

        raise ValueError(
            "\n"
            f"Could not uniquely resolve ground-truth ID "
            f"{relevant_id} for query "
            f"'{fusion_item.get('query')}'.\n\n"
            f"Dataset rows having id={relevant_id}: "
            f"{possible_rows}\n"
            f"Candidate matches: "
            f"{candidate_rows}\n\n"
            "The benchmark must not guess between duplicate "
            "dataset IDs."
        )

    return resolved_rows


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()
    print("=" * 70)
    print(
        "FAISS / DATASET ID ALIGNMENT CHECK"
    )
    print("=" * 70)

    # ======================================================
    # DATASET
    # ======================================================

    print()
    print("Dataset:")
    print(DATASET_PATH)

    data = load_data(
        str(DATASET_PATH)
    )

    print(
        "Dataset documents:",
        len(data),
    )

    # ======================================================
    # EMBEDDINGS
    # ======================================================

    print()
    print("Embeddings:")
    print(EMBEDDINGS_PATH)

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    print(
        "Embedding shape:",
        embeddings.shape,
    )

    print(
        "Embedding rows:",
        len(embeddings),
    )

    # ======================================================
    # BASIC ALIGNMENT
    # ======================================================

    print()
    print("-" * 70)
    print("BASIC DATASET / EMBEDDING ALIGNMENT")
    print("-" * 70)

    if len(data) == len(embeddings):

        print(
            "PASS: dataset rows == embedding rows"
        )

    else:

        print(
            "FAIL: dataset rows != embedding rows"
        )

        raise SystemExit(1)

    # ======================================================
    # DOCUMENT ID STRUCTURE
    # ======================================================

    print()
    print("-" * 70)
    print("DOCUMENT ID STRUCTURE")
    print("-" * 70)

    id_to_rows = build_id_to_rows(
        data
    )

    unique_ids = len(
        id_to_rows
    )

    duplicate_ids = {

        document_id: rows

        for document_id, rows
        in id_to_rows.items()

        if len(rows) > 1

    }

    print(
        "Unique dataset IDs:",
        unique_ids,
    )

    print(
        "Dataset IDs with duplicates:",
        len(duplicate_ids),
    )

    print()

    if duplicate_ids:

        print(
            "IMPORTANT: Dataset `id` is NOT a globally "
            "unique FAISS identifier."
        )

        print()

        for document_id in [
            6,
            25,
            108,
            114,
        ]:

            print(
                f"id={document_id} "
                f"-> FAISS rows="
                f"{id_to_rows.get(document_id, [])}"
            )

    else:

        print(
            "Dataset IDs are unique."
        )

    # ======================================================
    # SAMPLE ROWS
    # ======================================================

    print()
    print("-" * 70)
    print("DATASET ROW / ID / FAISS MAPPING")
    print("-" * 70)

    for row_index in range(
        min(15, len(data))
    ):

        document = data[row_index]

        print(
            f"FAISS row={row_index} | "
            f"dataset id={document.get('id')} | "
            f"source={document.get('source')}"
        )

    # ======================================================
    # FUSION DATASET
    # ======================================================

    print()
    print("-" * 70)
    print("GROUND-TRUTH RESOLUTION")
    print("-" * 70)

    with FUSION_DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        fusion_dataset = json.load(
            file
        )

    resolved_all = []

    for item in fusion_dataset:

        relevant_ids = item.get(
            "relevant_doc_ids",
            [],
        )

        if not relevant_ids:
            continue

        print()
        print(
            f"Query: {item['query']}"
        )

        print(
            "Original dataset IDs:",
            relevant_ids,
        )

        resolved_rows = resolve_ground_truth(
            data,
            item,
        )

        print(
            "Resolved FAISS rows:",
            resolved_rows,
        )

        for row in resolved_rows:

            document = data[row]

            print(
                f"  FAISS row {row} "
                f"-> dataset id {document.get('id')} "
                f"-> source {document.get('source')}"
            )

        resolved_all.append(
            (
                item["query"],
                resolved_rows,
            )
        )

    # ======================================================
    # FINAL
    # ======================================================

    print()
    print("=" * 70)
    print("ALIGNMENT RESULT")
    print("=" * 70)

    print(
        "PASS - dataset and embeddings are row-aligned."
    )

    print(
        "PASS - ground truth can be resolved to exact "
        "FAISS row IDs."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "FAISS search results must be evaluated using "
        "FAISS row indices."
    )

    print(
        "Dataset `id` values must NOT be compared directly "
        "against FAISS result indices."
    )

    print()
    print(
        "Alignment check complete."
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()