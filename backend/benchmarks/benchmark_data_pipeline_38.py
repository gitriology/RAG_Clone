"""Optimization #38 — PDF text aggregation benchmark.

The source recommendation for Optimization #38 is to accumulate page text
in a list and join once instead of repeatedly concatenating strings. This
benchmark isolates that aggregation operation so PDF rendering/extraction
time does not obscure the optimization. It is not an end-to-end RAG latency
benchmark.
"""

from __future__ import annotations

import time


def _old_string_concat(page_texts):
    text = ""
    for page_text in page_texts:
        if page_text:
            text += page_text + "\n\n"
    return text.strip()


def _list_join(page_texts):
    parts = []
    for page_text in page_texts:
        if page_text:
            parts.append(page_text)
    return "\n\n".join(parts).strip()


def _timed(fn, page_texts, repeats=5):
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(page_texts)
        samples.append(time.perf_counter() - start)
    return sum(samples) / len(samples), result


def main() -> None:
    # Large synthetic page-text payload to make allocation behavior visible.
    page_texts = [
        ("Synthetic PDF page text with stable content. " * 250).strip()
        for _ in range(500)
    ]

    old_time, old_text = _timed(_old_string_concat, page_texts)
    new_time, new_text = _timed(_list_join, page_texts)

    reduction = ((old_time - new_time) / old_time * 100.0) if old_time else 0.0

    print("Optimization #38 — PDF text aggregation")
    print("Synthetic pages: 500")
    print("Synthetic text per page: 250 repeated fragments")
    print(f"Text equivalence: {'identical' if old_text == new_text else 'different'}")
    print(f"Repeated string concatenation baseline: {old_time:.6f} s")
    print(f"List accumulation + one join: {new_time:.6f} s")
    print(f"Measured aggregation-time reduction: {reduction:.2f}%")
    print("Note: isolated text-aggregation benchmark; not end-to-end PDF or RAG latency.")


if __name__ == "__main__":
    main()
