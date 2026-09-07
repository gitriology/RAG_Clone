import re
from pathlib import Path

from backend.data_pipeline.run_pipeline import (
    _build_page_section_map,
    _source_timestamp,
    build_chunk_record,
)


def test_metadata_enhancement_adds_required_fields():
    pages = [
        {"page_number": 1, "text": "Introduction\nThis is content."},
        {"page_number": 2, "text": "More content."},
    ]
    sections = _build_page_section_map(pages)

    record = build_chunk_record(
        {
            "text": "This is content.",
            "chunk_index": 0,
            "page_start": 1,
            "page_end": 1,
            "word_count": 3,
        },
        document_id="doc123",
        numeric_id=7,
        source="example.pdf",
        source_path="data/raw/example.pdf",
        domain="general",
        document_type="document",
        source_file_hash="ignored-for-41",
        page_sections=sections,
        timestamp="2026-01-01T00:00:00+00:00",
    )

    for field in ("document_id", "page_number", "chunk_index", "section", "timestamp"):
        assert field in record

    assert record["document_id"] == "doc123"
    assert record["page_number"] == 1
    assert record["chunk_index"] == 0
    assert record["section"] == "Introduction"
    assert record["timestamp"] == "2026-01-01T00:00:00+00:00"

    # Existing metadata remains untouched.
    assert record["id"] == 7
    assert record["domain"] == "general"
    assert record["source"] == "example.pdf"
    assert record["type"] == "document"


def test_metadata_is_stable_and_page_span_is_preserved(tmp_path):
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"not-used-by-test")

    timestamp_1 = _source_timestamp(pdf)
    timestamp_2 = _source_timestamp(pdf)
    assert timestamp_1 == timestamp_2
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T.*\+00:00",
        timestamp_1,
    )

    pages = [
        {"page_number": 3, "text": "Section A\nFirst page."},
        {"page_number": 4, "text": "Continuation."},
    ]
    sections = _build_page_section_map(pages)

    record = build_chunk_record(
        {
            "text": "First page. Continuation.",
            "chunk_index": 2,
            "page_start": 3,
            "page_end": 4,
            "word_count": 4,
        },
        document_id="doc123",
        numeric_id=8,
        source="source.pdf",
        source_path="source.pdf",
        domain="general",
        document_type="document",
        source_file_hash="hash",
        page_sections=sections,
        timestamp=timestamp_1,
    )

    assert record["page_number"] == 3
    assert record["page_start"] == 3
    assert record["page_end"] == 4
    assert record["section"] == "Section A"


def test_missing_section_is_not_invented():
    pages = [
        {"page_number": 1, "text": "This is ordinary prose without a heading."},
    ]
    sections = _build_page_section_map(pages)

    record = build_chunk_record(
        {
            "text": "This is ordinary prose without a heading.",
            "chunk_index": 0,
            "page_start": 1,
            "page_end": 1,
        },
        document_id="doc",
        numeric_id=0,
        source="source.pdf",
        source_path="source.pdf",
        domain="general",
        document_type="document",
        source_file_hash="hash",
        page_sections=sections,
        timestamp="2026-01-01T00:00:00+00:00",
    )

    assert record["section"] is None
