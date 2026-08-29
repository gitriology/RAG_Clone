"""
Production PDF ingestion layer.

Goals
-----
1. Extract PDF text using layout-aware blocks instead of plain
   page.get_text("text").
2. Preserve reading order as much as possible.
3. Detect and separate multi-column layouts.
4. Remove image-only blocks.
5. Preserve page boundaries and provenance.
6. Detect repeated headers/footers later in the cleaning stage.
7. Remain backward compatible with:

       load_pdf(path)

   which returns a plain string.

For the production pipeline use:

       load_pdf(path, return_pages=True)

   which returns page records with layout/provenance metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import re

import fitz  # PyMuPDF


# ==========================================================
# CONFIGURATION
# ==========================================================

MIN_BLOCK_CHARS = 2
MIN_LINE_CHARS = 2

# Very small x/y differences caused by PDF rendering should
# not create separate columns.
X_CLUSTER_TOLERANCE = 18.0

# If the page contains a large amount of whitespace between
# two text regions, this is a strong indication of columns.
COLUMN_GAP_MIN = 45.0

# Ignore extremely small text blocks which are normally page
# decorations / artefacts.
MIN_BLOCK_WIDTH = 4.0
MIN_BLOCK_HEIGHT = 3.0


# ==========================================================
# DATA TYPES
# ==========================================================

@dataclass
class TextBlock:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    block_no: int
    page_number: int
    block_type: int = 0

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0


# ==========================================================
# BASIC TEXT NORMALIZATION
# ==========================================================

def _normalize_block_text(text: str) -> str:
    """
    Normalize only extraction-level whitespace.

    Important:
        This function does NOT perform aggressive semantic
        cleaning. That belongs in clean_text.py.
    """

    if not text:
        return ""

    text = text.replace("\x00", " ")

    # Normalize CR/LF variants.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove invisible control characters except newline/tab.
    text = "".join(
        ch
        for ch in text
        if ch == "\n"
        or ch == "\t"
        or ord(ch) >= 32
    )

    # Trim every physical line.
    lines = []

    for line in text.split("\n"):

        line = re.sub(
            r"[ \t]+",
            " ",
            line,
        ).strip()

        if line:
            lines.append(line)

    return "\n".join(lines).strip()


# ==========================================================
# BLOCK EXTRACTION
# ==========================================================

def _extract_blocks(
    page: fitz.Page,
    page_number: int,
) -> List[TextBlock]:
    """
    Extract text blocks with coordinates.

    PyMuPDF's block extraction is much safer for documents
    containing columns/tables than get_text("text").
    """

    raw_blocks = page.get_text(
        "blocks",
        sort=False,
    )

    blocks: List[TextBlock] = []

    for block in raw_blocks:

        if len(block) < 5:
            continue

        x0, y0, x1, y1, text = block[:5]

        block_no = (
            int(block[5])
            if len(block) > 5
            and isinstance(block[5], (int, float))
            else len(blocks)
        )

        block_type = (
            int(block[6])
            if len(block) > 6
            and isinstance(block[6], (int, float))
            else 0
        )

        # block_type == 1 is normally an image.
        if block_type == 1:
            continue

        text = _normalize_block_text(
            str(text or "")
        )

        if not text:
            continue

        if len(text) < MIN_BLOCK_CHARS:
            continue

        if (
            abs(x1 - x0)
            < MIN_BLOCK_WIDTH
            or abs(y1 - y0)
            < MIN_BLOCK_HEIGHT
        ):
            continue

        blocks.append(
            TextBlock(
                text=text,
                x0=float(x0),
                y0=float(y0),
                x1=float(x1),
                y1=float(y1),
                block_no=block_no,
                page_number=page_number,
                block_type=block_type,
            )
        )

    return blocks


# ==========================================================
# COLUMN DETECTION
# ==========================================================

def _overlap_ratio(
    a: TextBlock,
    b: TextBlock,
) -> float:
    """
    Horizontal overlap ratio.
    """

    overlap = max(
        0.0,
        min(a.x1, b.x1)
        - max(a.x0, b.x0),
    )

    width = min(
        a.width,
        b.width,
    )

    if width <= 0:
        return 0.0

    return overlap / width


def _estimate_columns(
    blocks: Sequence[TextBlock],
    page_width: float,
) -> List[List[TextBlock]]:
    """
    Estimate logical reading columns.

    This intentionally avoids assuming that every PDF has
    exactly two columns.

    Strategy
    --------
    1. Sort blocks by left edge.
    2. Cluster nearby left edges.
    3. Validate clusters against actual geometry.
    4. Reject fake columns caused by tables/forms.
    """

    if len(blocks) <= 1:
        return [list(blocks)]

    sorted_blocks = sorted(
        blocks,
        key=lambda b: (
            b.x0,
            b.y0,
        ),
    )

    clusters: List[List[TextBlock]] = []

    for block in sorted_blocks:

        placed = False

        for cluster in clusters:

            mean_x = sum(
                b.x0
                for b in cluster
            ) / len(cluster)

            if abs(
                block.x0 - mean_x
            ) <= X_CLUSTER_TOLERANCE:

                cluster.append(block)
                placed = True
                break

        if not placed:

            clusters.append(
                [block]
            )

    # If there are many tiny clusters, this is likely a table
    # or irregular layout rather than columns.
    if len(clusters) <= 1:
        return [list(blocks)]

    if len(clusters) > 4:
        return [list(blocks)]

    # Sort by left edge.
    clusters.sort(
        key=lambda cluster:
        min(b.x0 for b in cluster)
    )

    # Merge clusters that are not genuinely separated.
    merged: List[List[TextBlock]] = []

    for cluster in clusters:

        if not merged:
            merged.append(cluster)
            continue

        previous = merged[-1]

        previous_right = max(
            b.x1
            for b in previous
        )

        current_left = min(
            b.x0
            for b in cluster
        )

        gap = current_left - previous_right

        if gap < COLUMN_GAP_MIN:

            merged[-1].extend(
                cluster
            )

        else:

            merged.append(
                cluster
            )

    # A genuine two-column page normally has columns occupying
    # meaningful portions of the page width.
    if len(merged) == 2:

        left_width = max(
            b.x1
            for b in merged[0]
        ) - min(
            b.x0
            for b in merged[0]
        )

        right_width = max(
            b.x1
            for b in merged[1]
        ) - min(
            b.x0
            for b in merged[1]
        )

        if (
            left_width < page_width * 0.20
            or right_width < page_width * 0.20
        ):
            return [list(blocks)]

    return merged


# ==========================================================
# BLOCK READING ORDER
# ==========================================================

def _sort_blocks_in_column(
    blocks: Sequence[TextBlock],
) -> List[TextBlock]:
    """
    Sort blocks in natural top-to-bottom order.

    Small vertical overlaps are resolved using x position.
    """

    return sorted(
        blocks,
        key=lambda b: (
            round(b.y0 / 4.0),
            b.x0,
            b.y1,
        ),
    )


def _sort_page_blocks(
    blocks: Sequence[TextBlock],
    page_width: float,
) -> List[TextBlock]:
    """
    Determine reading order for a page.

    Single column:
        top -> bottom

    Multiple columns:
        left column top -> bottom
        right column top -> bottom

    This prevents the classic PDF failure:

        left line
        right line
        left line
        right line
    """

    if not blocks:
        return []

    columns = _estimate_columns(
        blocks,
        page_width,
    )

    if len(columns) <= 1:

        return _sort_blocks_in_column(
            blocks
        )

    ordered: List[TextBlock] = []

    for column in columns:

        ordered.extend(
            _sort_blocks_in_column(
                column
            )
        )

    return ordered


# ==========================================================
# PAGE EXTRACTION
# ==========================================================

def extract_page(
    page: fitz.Page,
    page_number: int,
) -> Dict[str, Any]:
    """
    Extract one PDF page into a structured record.
    """

    page_width = float(
        page.rect.width
    )

    page_height = float(
        page.rect.height
    )

    blocks = _extract_blocks(
        page,
        page_number,
    )

    ordered = _sort_page_blocks(
        blocks,
        page_width,
    )

    text_parts = []

    for block in ordered:

        text = block.text.strip()

        if not text:
            continue

        text_parts.append(text)

    text = "\n\n".join(
        text_parts
    ).strip()

    return {
        "page_number": page_number,
        "width": page_width,
        "height": page_height,
        "text": text,
        "blocks": [
            asdict(block)
            for block in ordered
        ],
    }


# ==========================================================
# DOCUMENT EXTRACTION
# ==========================================================

def load_pdf_pages(
    file_path: str | Path,
) -> List[Dict[str, Any]]:
    """
    Production PDF loader.

    Returns one structured dictionary per page.
    """

    file_path = Path(
        file_path
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"PDF path is not a file: {file_path}"
        )

    pages: List[Dict[str, Any]] = []

    document = fitz.open(
        str(file_path)
    )

    try:

        if document.needs_pass:
            raise RuntimeError(
                f"PDF is password protected: "
                f"{file_path}"
            )

        for page_index in range(
            len(document)
        ):

            page = document[
                page_index
            ]

            page_data = extract_page(
                page,
                page_number=page_index + 1,
            )

            pages.append(
                page_data
            )

    finally:

        document.close()

    return pages


# ==========================================================
# BACKWARD COMPATIBLE LOADER
# ==========================================================

def load_pdf(
    file_path: str | Path,
    return_pages: bool = False,
):
    """
    Backward-compatible public API.

    load_pdf(path)
        -> plain text string

    load_pdf(path, return_pages=True)
        -> list of page dictionaries
    """

    pages = load_pdf_pages(
        file_path
    )

    if return_pages:
        return pages

    return "\n\n".join(
        page["text"]
        for page in pages
        if page.get("text")
    ).strip()


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Layout-aware PDF text extraction."
    )

    parser.add_argument(
        "pdf",
        help="PDF file to inspect.",
    )

    args = parser.parse_args()

    pages = load_pdf(
        args.pdf,
        return_pages=True,
    )

    print(
        f"Pages: {len(pages)}"
    )

    for page in pages:

        print()
        print(
            "=" * 70
        )

        print(
            f"PAGE {page['page_number']}"
        )

        print(
            "=" * 70
        )

        print(
            page["text"]
        )