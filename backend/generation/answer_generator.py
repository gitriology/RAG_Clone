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
# BUILD ANSWER
# ==========================================================

def generate_answer(
    query,
    documents,
    max_sentences=3
):
    """
    Parameters
    ----------
    query : str

    documents : list
        Output of reranker.

    Returns
    -------
    str
    """

    all_sentences = []

    for doc in documents:

        all_sentences.extend(

            split_sentences(

                doc["text"]

            )

        )

    all_sentences = remove_duplicates(
        all_sentences
    )

    scored = score_sentences(
        query,
        all_sentences
    )

    selected = [

        sentence

        for sentence, score in scored[:max_sentences]

    ]

    answer = " ".join(selected)

    answer = answer.strip()

    if not answer.endswith("."):

        answer += "."

    print()

    print("=" * 60)

    print("Answer Generator")

    print("=" * 60)

    print(f"Candidate Sentences : {len(all_sentences)}")

    print(f"Selected Sentences  : {len(selected)}")

    if scored:

        print(
            f"Best Similarity     : {scored[0][1]:.4f}"
        )

    print("=" * 60)

    return answer