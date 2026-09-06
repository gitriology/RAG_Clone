from backend.data_pipeline.cleaning.clean_text import clean_pages, clean_text


def _page(page_number, text, blocks):
    return {
        "page_number": page_number,
        "width": 600.0,
        "height": 800.0,
        "text": text,
        "blocks": blocks,
    }


def test_page_number_removed_when_metadata_and_edge_geometry_agree():
    pages = [
        _page(1, "Report title\nBody content\n1", [
            {"text": "Report title", "y0": 20, "y1": 40},
            {"text": "Body content", "y0": 200, "y1": 230},
            {"text": "1", "y0": 760, "y1": 780},
        ]),
        _page(2, "Report title\nMore content\n2", [
            {"text": "Report title", "y0": 20, "y1": 40},
            {"text": "More content", "y0": 200, "y1": 230},
            {"text": "2", "y0": 760, "y1": 780},
        ]),
        _page(3, "Report title\nThird content\n3", [
            {"text": "Report title", "y0": 20, "y1": 40},
            {"text": "Third content", "y0": 200, "y1": 230},
            {"text": "3", "y0": 760, "y1": 780},
        ]),
    ]
    cleaned = clean_pages(pages)
    assert cleaned[0]["text"] == "Body content"
    assert cleaned[1]["text"] == "More content"
    assert cleaned[0]["page_number"] == 1


def test_legitimate_body_page_reference_is_preserved():
    page = _page(
        7,
        "According to Page 12, the result is valid.\n42",
        [
            {"text": "According to Page 12, the result is valid.", "y0": 250, "y1": 290},
            {"text": "42", "y0": 300, "y1": 320},
        ],
    )
    cleaned = clean_pages([page])
    assert "Page 12" in cleaned[0]["text"]
    assert "42" in cleaned[0]["text"]


def test_plain_text_api_retains_generic_page_number_cleaning():
    cleaned = clean_text("Heading\nPage 12\nBody")
    assert "Page 12" not in cleaned
    assert "Heading" in cleaned and "Body" in cleaned
