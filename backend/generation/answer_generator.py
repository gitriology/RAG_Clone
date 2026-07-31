"""
Phase 10 : Answer Generator

Generates a concise evidence-grounded answer from the
top retrieved documents.

Pipeline

Query
    ↓
Sentence Extraction
    ↓
Sentence Embedding
    ↓
Similarity Scoring
    ↓
Duplicate Removal
    ↓
Top Sentence Selection
    ↓
Final Answer
"""

import re
import numpy as np
from sentence_transformers import util

from backend.retrieval.dense.embedder import model


# ==========================================================
# CLEAN TEXT
# ==========================================================

def clean_text(text: str) -> str:

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================================
# SPLIT INTO SENTENCES
# ==========================================================

def split_sentences(text: str):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        clean_text(text)
    )

    return [

        s.strip()

        for s in sentences

        if len(s.split()) >= 6

    ]


# ==========================================================
# REMOVE DUPLICATE SENTENCES
# ==========================================================

def remove_duplicates(sentences):

    unique = []

    seen = set()

    for sentence in sentences:

        key = sentence.lower()

        if key not in seen:

            seen.add(key)

            unique.append(sentence)

    return unique


# ==========================================================
# SCORE SENTENCES
# ==========================================================

def score_sentences(query, sentences):

    if not sentences:

        return []

    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    sentence_embeddings = model.encode(
        sentences,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    similarities = util.cos_sim(
        query_embedding,
        sentence_embeddings
    )[0]

    scored = []

    for sentence, score in zip(
        sentences,
        similarities
    ):

        scored.append(

            (

                sentence,

                float(score)

            )

        )

    scored.sort(

        key=lambda x: x[1],

        reverse=True

    )

    return scored


# ==========================================================
# GENERATE ANSWER
# ==========================================================

def generate_answer(
    query,
    documents,
    max_sentences=3,
):
    """
    Phase 10.1

    Performs semantic sentence selection using BGE embeddings.

    Returns a structured object for downstream
    Evidence Fusion (Phase 10.2).
    """

    # ------------------------------------------------------
    # Collect candidate sentences
    # ------------------------------------------------------

    all_sentences = []

    for doc in documents:

        all_sentences.extend(

            split_sentences(
                doc["text"]
            )

        )

    # ------------------------------------------------------
    # Remove duplicates
    # ------------------------------------------------------

    all_sentences = remove_duplicates(
        all_sentences
    )

    # ------------------------------------------------------
    # Semantic ranking
    # ------------------------------------------------------

    scored = score_sentences(
        query,
        all_sentences
    )

    # ------------------------------------------------------
    # Select Top-N
    # ------------------------------------------------------

    selected = scored[:max_sentences]

    answer = " ".join(

        sentence

        for sentence, _ in selected

    ).strip()

    if answer and not answer.endswith("."):

        answer += "."

    # ------------------------------------------------------
    # Structured Evidence
    # ------------------------------------------------------

    evidence = []

    for rank, (sentence, score) in enumerate(
        selected,
        start=1,
    ):

        evidence.append(

            {

                "rank": rank,

                "text": sentence,

                "similarity": round(
                    float(score),
                    4,
                ),

            }

        )

    # ------------------------------------------------------
    # Debug
    # ------------------------------------------------------

    print()

    print("=" * 60)

    print("Phase 10.1 : Semantic Sentence Selection")

    print("=" * 60)

    print(
        f"Candidate Sentences : {len(all_sentences)}"
    )

    print(
        f"Selected Sentences  : {len(evidence)}"
    )

    if evidence:

        print(
            f"Best Similarity     : {evidence[0]['similarity']:.4f}"
        )

    print("=" * 60)

    # ------------------------------------------------------
    # Return Structured Object
    # ------------------------------------------------------

    return {

        "answer": answer,

        "selected_sentences": evidence,

        "candidate_sentences": len(
            all_sentences
        ),

        "selected_count": len(
            evidence
        ),

        "best_similarity": (

            evidence[0]["similarity"]

            if evidence

            else 0.0

        ),

    }