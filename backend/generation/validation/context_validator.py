"""
Context Validation

Quality-aware document validation for the production pipeline.

Changes:
- shared validation model
- batch document encoding
- normalized embeddings
- sentence-level best-match support
- lightweight text-quality scoring
- no destructive document filtering
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from sentence_transformers import util

from backend.models.model_registry import ModelRegistry
from backend.generation.validation.embedding_cache import ValidationEmbeddingCache


BROKEN_PATTERNS = (
    "document contains references",
    "supporting documents throughout",
    "the current •",
    "• and document",
    "this icon indicates",
    "users seeking practical advice",
    "euro.who.int",
)


def _text_quality(text: str) -> float:
    if not text:
        return 0.0

    s = str(text).strip()
    words = s.split()

    if len(words) < 5:
        return 0.0

    score = 1.0
    lower = s.lower()

    for pattern in BROKEN_PATTERNS:
        if pattern in lower:
            score -= 0.20

    if "•" in s:
        # Bullets are not bad by themselves, but excessive bullets
        # often indicate raw PDF extraction.
        if s.count("•") >= 3:
            score -= 0.20

    if len(words) > 450:
        score -= 0.10

    return max(0.0, min(score, 1.0))


def _extract_sentences(text: str) -> List[str]:
    if not text:
        return []

    text = re.sub(r"https?://\S+|www\.\S+", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()

    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9•])|\s*•\s*",
        text,
    )

    return [
        p.strip()
        for p in parts
        if len(p.split()) >= 5
    ]


def validate_context(
    query,
    documents,
    threshold=0.30,
    embedding_cache: ValidationEmbeddingCache | None = None,
):

    if not documents:
        return []

    model = ModelRegistry.get_validation_model()
    embedding_cache = embedding_cache or ValidationEmbeddingCache(model=model)

    start_time = time.perf_counter()

    query_embedding = embedding_cache.encode([query])

    document_texts = [
        doc.get("text", "")
        for doc in documents
    ]

    doc_embeddings = embedding_cache.encode(document_texts)

    document_scores = util.cos_sim(
        query_embedding,
        doc_embeddings,
    )[0]

    # ------------------------------------------------------
    # Sentence-level support.
    #
    # A long document can contain the answer even when its
    # whole-document embedding is diluted by unrelated text.
    # ------------------------------------------------------

    sentence_candidates: List[str] = []
    sentence_owner: List[int] = []

    for index, text in enumerate(document_texts):
        sentences = _extract_sentences(text)

        # Limit work per document while keeping the beginning,
        # middle and end represented.
        if len(sentences) > 120:
            step = max(1, len(sentences) // 120)
            sentences = sentences[::step][:120]

        for sentence in sentences:
            sentence_candidates.append(sentence)
            sentence_owner.append(index)

    best_sentence_scores = [0.0] * len(documents)

    if sentence_candidates:
        sentence_embeddings = embedding_cache.encode(sentence_candidates)

        sentence_scores = util.cos_sim(
            query_embedding,
            sentence_embeddings,
        )[0]

        for owner, score in zip(
            sentence_owner,
            sentence_scores,
        ):
            value = float(score.item())

            if value > best_sentence_scores[owner]:
                best_sentence_scores[owner] = value

    # ------------------------------------------------------
    # Attach quality-aware context score.
    # ------------------------------------------------------

    for index, doc in enumerate(documents):

        whole_score = float(
            document_scores[index].item()
        )

        best_sentence = best_sentence_scores[index]

        quality = _text_quality(
            document_texts[index]
        )

        # Sentence support gets enough weight to rescue a long
        # document whose overall embedding is diluted.
        context_score = (
            0.55 * whole_score
            + 0.35 * best_sentence
            + 0.10 * quality
        )

        # Do not let low-quality extracted text dominate.
        if quality < 0.40:
            context_score *= 0.60

        doc["context_score"] = float(
            max(0.0, min(context_score, 1.0))
        )

        doc["context_whole_score"] = whole_score
        doc["context_sentence_score"] = best_sentence
        doc["context_quality"] = quality

    elapsed = time.perf_counter() - start_time

    print(
        f"[Context Validation] Documents: {len(documents)}"
    )

    print(
        f"[Context Validation] Embedding + similarity time: "
        f"{elapsed:.4f}s"
    )

    print(
        "[Context Validation] Model ID:",
        id(model),
    )

    cache_info = embedding_cache.info()
    print(
        "[Optimization #30] Validation embedding cache:",
        cache_info,
    )

    return documents
