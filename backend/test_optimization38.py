from pathlib import Path
import sys

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.data_pipeline.ingestion.load_pdf import (  # noqa: E402
    load_pdf,
    load_pdf_pages,
    load_pdf_text,
)


def _make_pdf(path: Path) -> None:
    doc = fitz.open()
    for page_no in range(1, 4):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {page_no}")
        page.insert_text(
            (72, 110),
            "This is a small deterministic document used to verify the optimized plain-text ingestion path.",
        )
    doc.save(path)
    doc.close()


def test_fast_text_path_preserves_page_text_and_backward_api(tmp_path):
    pdf = tmp_path / "sample.pdf"
    _make_pdf(pdf)

    fast = load_pdf_text(pdf)
    compat = load_pdf(pdf)
    pages = load_pdf(pdf, return_pages=True)

    assert fast.replace("\n\n", "\n") == compat.replace("\n\n", "\n")
    assert "Page 1" in fast
    assert "Page 3" in fast
    assert len(pages) == 3
    assert [page["page_number"] for page in pages] == [1, 2, 3]


def test_page_aware_path_retains_provenance(tmp_path):
    pdf = tmp_path / "sample.pdf"
    _make_pdf(pdf)

    pages = load_pdf_pages(pdf)

    assert all("page_number" in page for page in pages)
    assert all("text" in page for page in pages)
    assert all("blocks" in page for page in pages)
    assert all(page["blocks"] for page in pages)


def test_fast_path_avoids_page_block_materialization(monkeypatch, tmp_path):
    pdf = tmp_path / "sample.pdf"
    _make_pdf(pdf)

    import backend.data_pipeline.ingestion.load_pdf as loader

    def fail_extract_page(*args, **kwargs):
        raise AssertionError("fast text path should not build layout blocks")

    monkeypatch.setattr(loader, "extract_page", fail_extract_page)
    text = loader.load_pdf_text(pdf)

    assert "Page 1" in text
