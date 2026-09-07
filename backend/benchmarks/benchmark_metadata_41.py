"""
Optimization #41 benchmark.

Measures metadata enrichment overhead and completeness on synthetic
page/chunk records. This is intentionally isolated from embedding,
FAISS, BM25, reranking, and Evidence State timings.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from backend.data_pipeline.run_pipeline import (
    _build_page_section_map,
    build_chunk_record,
)


def _baseline_record(chunk, document_id, numeric_id, source, source_path, domain, document_type):
    return {
        "id": numeric_id,
        "text": chunk["text"],
        "domain": domain,
        "source": source,
        "type": document_type,
        "document_id": document_id,
        "source_path": source_path,
        "chunk_index": int(chunk["chunk_index"]),
        "page_start": chunk["page_start"],
        "page_end": chunk["page_end"],
        "word_count": len(chunk["text"].split()),
    }


def main() -> None:
    page_count = 1000
    chunks_per_page = 3
    pages = [
        {
            "page_number": page,
            "text": (
                f"Section {page}\n"
                "This is representative document content for metadata benchmarking."
            ),
        }
        for page in range(1, page_count + 1)
    ]

    chunks = []
    for page in pages:
        for local_index in range(chunks_per_page):
            chunks.append(
                {
                    "text": (
                        f"Section {page['page_number']} "
                        f"chunk {local_index} representative content."
                    ),
                    "chunk_index": (
                        (page["page_number"] - 1) * chunks_per_page
                        + local_index
                    ),
                    "page_start": page["page_number"],
                    "page_end": page["page_number"],
                }
            )

    document_id = "synthetic-document"
    source = "synthetic.pdf"
    source_path = "data/raw/synthetic.pdf"
    domain = "general"
    document_type = "document"
    timestamp = datetime.now(timezone.utc).isoformat()

    start = time.perf_counter()
    baseline = [
        _baseline_record(
            chunk,
            document_id,
            i,
            source,
            source_path,
            domain,
            document_type,
        )
        for i, chunk in enumerate(chunks)
    ]
    baseline_ms = (time.perf_counter() - start) * 1000

    page_sections = _build_page_section_map(pages)

    start = time.perf_counter()
    optimized = [
        build_chunk_record(
            chunk,
            document_id=document_id,
            numeric_id=i,
            source=source,
            source_path=source_path,
            domain=domain,
            document_type=document_type,
            source_file_hash="synthetic",
            page_sections=page_sections,
            timestamp=timestamp,
        )
        for i, chunk in enumerate(chunks)
    ]
    optimized_ms = (time.perf_counter() - start) * 1000

    required = ("document_id", "page_number", "chunk_index", "section", "timestamp")
    complete = sum(
        all(record.get(field) is not None for field in required)
        for record in optimized
    )
    completeness = complete / len(optimized) if optimized else 0.0

    print("Optimization #41 metadata benchmark")
    print(f"Pages: {page_count}")
    print(f"Chunks: {len(chunks)}")
    print(f"Required-field completeness: {completeness:.3f}")
    print(f"Baseline metadata build: {baseline_ms:.3f} ms")
    print(f"Optimized metadata build: {optimized_ms:.3f} ms")
    print(
        f"Metadata enrichment overhead: "
        f"{optimized_ms - baseline_ms:.3f} ms"
    )
    print(
        "Note: #41 is a provenance/retrieval-quality enhancement; "
        "the benchmark should not be interpreted as an end-to-end speedup."
    )


if __name__ == "__main__":
    main()
