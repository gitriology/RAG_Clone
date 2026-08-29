from pathlib import Path
import json
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


# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("FAISS / GROUND-TRUTH ID ALIGNMENT CHECK")
print("=" * 60)

print()
print("Dataset:")
print(DATASET_PATH)

data = load_data(
    str(DATASET_PATH)
)

print(
    "Dataset documents:",
    len(data)
)


# ==========================================================
# LOAD EMBEDDINGS
# ==========================================================

print()
print("Embeddings:")
print(EMBEDDINGS_PATH)

embeddings = np.load(
    EMBEDDINGS_PATH
)

print(
    "Embedding shape:",
    embeddings.shape
)

print(
    "Embedding rows:",
    len(embeddings)
)


# ==========================================================
# CHECK BASIC ALIGNMENT
# ==========================================================

print()
print("-" * 60)
print("BASIC ALIGNMENT")
print("-" * 60)

print(
    "Dataset length:",
    len(data)
)

print(
    "Embedding rows:",
    len(embeddings)
)

if len(data) == len(embeddings):
    print(
        "RESULT: PASS - same number of dataset rows and embeddings"
    )
else:
    print(
        "RESULT: FAIL - dataset rows and embeddings differ"
    )


# ==========================================================
# INSPECT DOCUMENT IDs
# ==========================================================

print()
print("-" * 60)
print("DOCUMENT ID STRUCTURE")
print("-" * 60)

for i in range(
    min(15, len(data))
):

    document = data[i]

    print(
        f"row={i} | "
        f"doc_id={document.get('doc_id', '<NO doc_id FIELD>')} | "
        f"keys={list(document.keys())}"
    )


# ==========================================================
# TEST GROUND-TRUTH IDS
# ==========================================================

ground_truth_ids = [
    6,
    25,
    108,
    114,
]

print()
print("-" * 60)
print("GROUND-TRUTH ID CHECK")
print("-" * 60)

for doc_id in ground_truth_ids:

    print()
    print(
        f"Checking ground-truth doc_id={doc_id}"
    )

    # ------------------------------------------------------
    # Interpretation A:
    # doc_id is directly the dataset/FAISS row number
    # ------------------------------------------------------

    if (
        isinstance(doc_id, int)
        and 0 <= doc_id < len(data)
    ):

        document = data[doc_id]

        print(
            "  As row index:"
        )

        print(
            f"    dataset[{doc_id}] exists"
        )

        print(
            f"    dataset[{doc_id}].doc_id = "
            f"{document.get('doc_id', '<NO doc_id>')}"
        )

        print(
            f"    embedding[{doc_id}] exists = YES"
        )

    else:

        print(
            "  As row index: INVALID"
        )


    # ------------------------------------------------------
    # Interpretation B:
    # doc_id is a field inside the dataset
    # ------------------------------------------------------

    matching_rows = []

    for row_index, document in enumerate(data):

        stored_id = document.get(
            "doc_id"
        )

        if stored_id == doc_id:
            matching_rows.append(
                row_index
            )

        elif str(stored_id) == str(doc_id):
            matching_rows.append(
                row_index
            )

    print(
        "  As document ID field:"
    )

    if matching_rows:

        print(
            f"    MATCHING DATASET ROWS: "
            f"{matching_rows}"
        )

    else:

        print(
            "    NO MATCHING doc_id FIELD"
        )


# ==========================================================
# SPECIFIC CHECK FOR doc_id = 6
# ==========================================================

print()
print("=" * 60)
print("DETAILED CHECK: doc_id = 6")
print("=" * 60)

if len(data) > 6:

    row_6 = data[6]

    print()
    print(
        "dataset[6]:"
    )

    print(
        json.dumps(
            row_6,
            indent=2,
            ensure_ascii=False,
        )[:3000]
    )

    print()
    print(
        "embedding[6] shape:",
        embeddings[6].shape
    )

else:

    print(
        "Dataset does not contain row 6."
    )


# ==========================================================
# FIND ACTUAL ROW FOR doc_id = 6
# ==========================================================

print()
print("-" * 60)
print("SEARCHING FOR STORED doc_id = 6")
print("-" * 60)

matches = []

for row_index, document in enumerate(data):

    stored_id = document.get(
        "doc_id"
    )

    if (
        stored_id == 6
        or str(stored_id) == "6"
    ):

        matches.append(
            row_index
        )

if matches:

    print(
        "doc_id=6 appears at dataset row(s):",
        matches
    )

    for row_index in matches:

        print()
        print(
            f"Row {row_index}:"
        )

        print(
            json.dumps(
                data[row_index],
                indent=2,
                ensure_ascii=False,
            )[:2000]
        )

else:

    print(
        "doc_id=6 was NOT found as a document field."
    )


# ==========================================================
# FINAL INTERPRETATION
# ==========================================================

print()
print("=" * 60)
print("INTERPRETATION")
print("=" * 60)

if len(data) != len(embeddings):

    print(
        "WARNING:"
    )

    print(
        "Dataset and embedding counts differ."
    )

elif (
    len(data) > 6
    and (
        data[6].get("doc_id") == 6
        or str(data[6].get("doc_id")) == "6"
    )
):

    print(
        "PASS:"
    )

    print(
        "dataset row 6 has doc_id=6."
    )

    print(
        "Therefore doc_id=6 appears to align"
        " directly with FAISS row 6."
    )

else:

    print(
        "IMPORTANT:"
    )

    print(
        "ground-truth doc_id=6 does NOT directly"
        " appear to be the document ID of dataset row 6."
    )

    print(
        "A document-ID -> FAISS-row mapping may be required."
    )

print()
print("=" * 60)
print("CHECK COMPLETE")
print("=" * 60)