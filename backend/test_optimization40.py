from __future__ import annotations

from backend.data_pipeline.chunking.chunker import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    MAX_CHUNK_WORDS,
    chunk_text,
)
from backend.benchmarks.benchmark_chunking_40 import (
    CONFIGS,
    SimpleBM25,
    answer_support_accuracy,
    context_relevance,
)


def test_experimental_configurations_are_supported_without_clamping():
    assert CONFIGS == ((200, 40), (300, 60), (400, 80), (500, 100))
    assert MAX_CHUNK_WORDS >= 500

    text = " ".join(
        f"Sentence {i} contains enough retrieval content for testing."
        for i in range(260)
    )

    counts = []
    for size, overlap in CONFIGS:
        chunks = chunk_text(text, chunk_size=size, overlap=overlap)
        assert chunks
        counts.append(len(chunks))

    # Distinct target sizes must produce distinct chunking behaviour.
    assert len(set(counts)) == len(counts)


def test_measured_winner_is_the_production_default():
    assert (DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP) == (500, 100)


def test_benchmark_metrics_are_bounded():
    assert 0.0 <= context_relevance(
        "machine learning",
        ["machine learning learns from data", "unrelated text"],
    ) <= 1.0

    item = {
        "gold_texts": ["machine learning learns patterns from data"],
    }
    assert answer_support_accuracy(
        item,
        ["machine learning learns patterns from data"],
    ) == 1.0

    corpus = [
        "machine learning learns patterns from data",
        "satellite communication uses radio links",
        "constitutional law defines government powers",
    ]
    index = SimpleBM25(corpus)
    assert index.top_indices("What is machine learning?", 2)[0] == 0
