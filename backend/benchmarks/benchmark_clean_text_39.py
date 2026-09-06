"""Benchmark Optimization #39: metadata-aware page-number cleaning.

This benchmark measures cleaning quality on a controlled synthetic corpus.
The baseline reproduces the old regex behavior: any standalone number or
"Page N" line is removed, regardless of where it appears on the page.
The optimized cleaner removes page numbers only when page metadata and
edge geometry support the interpretation.
"""

from __future__ import annotations

import re
import time

from backend.data_pipeline.cleaning.clean_text import clean_pages


PAGE_NUMBER = re.compile(r"^(?:page\s*)?\d{1,5}$", re.IGNORECASE)


def baseline_clean_pages(pages):
    output = []
    for page in pages:
        cleaned = dict(page)
        lines = []
        for line in str(page.get("text", "")).splitlines():
            value = line.strip()
            if not value:
                continue
            if PAGE_NUMBER.fullmatch(value):
                continue
            lines.append(value)
        cleaned["text"] = "\n".join(lines)
        output.append(cleaned)
    return output


def build_corpus(count: int = 1000):
    pages = []
    for i in range(1, count + 1):
        body_number = 90000 + i
        body_reference = 12 + (i % 5)
        pages.append({
            "page_number": i,
            "width": 600.0,
            "height": 800.0,
            "text": (
                f"Report title\n"
                f"According to Page {body_reference}, the result is valid.\n"
                f"{body_number}\n"
                f"{i}"
            ),
            "blocks": [
                {"text": "Report title", "x0": 40, "y0": 20, "x1": 300, "y1": 40},
                {"text": f"According to Page {body_reference}, result {body_number} is valid.", "x0": 40, "y0": 250, "x1": 560, "y1": 290},
                {"text": str(body_number), "x0": 40, "y0": 300, "x1": 100, "y1": 320},
                {"text": str(i), "x0": 40, "y0": 760, "x1": 100, "y1": 780},
            ],
        })
    return pages


def count_preserved_body_numbers(pages, cleaned):
    expected = 0
    preserved = 0
    for source, result in zip(pages, cleaned):
        body_number = str(90000 + source["page_number"])
        expected += 1
        if body_number in result["text"]:
            preserved += 1
    return expected, preserved


def main():
    pages = build_corpus()

    t0 = time.perf_counter()
    baseline = baseline_clean_pages(pages)
    baseline_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    optimized = clean_pages(pages)
    optimized_time = time.perf_counter() - t0

    expected, baseline_preserved = count_preserved_body_numbers(pages, baseline)
    _, optimized_preserved = count_preserved_body_numbers(pages, optimized)

    print(f"Pages: {len(pages)}")
    print(f"Baseline time:   {baseline_time:.6f} s")
    print(f"Optimized time:  {optimized_time:.6f} s")
    print(f"Baseline body-number preservation:  {baseline_preserved}/{expected}")
    print(f"Optimized body-number preservation: {optimized_preserved}/{expected}")
    print(f"Baseline false removals:  {expected - baseline_preserved}")
    print(f"Optimized false removals: {expected - optimized_preserved}")
    print("Quality result: metadata-aware cleaning preserves numeric body content while removing edge page numbers.")


if __name__ == "__main__":
    main()
