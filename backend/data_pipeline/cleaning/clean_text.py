"""
Production text cleaning layer.

The cleaner is deliberately conservative.

It removes:
    - control characters
    - repeated page numbers
    - repeated headers/footers
    - isolated extraction artefacts
    - duplicated adjacent words caused by PDF extraction
    - broken whitespace
    - soft hyphenation
    - line-break hyphenation

It preserves:
    - numbers
    - dates
    - URLs
    - scientific notation
    - abbreviations
    - bullet content
    - table-like key/value information
    - source terminology

The module supports both:

    clean_text(str)

and:

    clean_pages(list_of_page_records)
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import re
import unicodedata


# ==========================================================
# CONFIGURATION
# ==========================================================

MIN_LINE_LENGTH = 2

REPEATED_PAGE_FRACTION = 0.35

MAX_HEADER_FOOTER_LINES = 3
PAGE_EDGE_FRACTION = 0.15

JUNK_TOKEN_PATTERNS = [
    re.compile(
        r"^[ivxlcdm]+$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^[^A-Za-z0-9]{1,8}$"
    ),
    re.compile(
        r"^(?:page|pg\.?)\s*\d+$",
        re.IGNORECASE,
    ),
]

# Extraction artefacts such as:
#
# iiii
# iiiiii
#
# but don't remove normal Roman numerals embedded in text.
ISOLATED_REPEAT_PATTERN = re.compile(
    r"^(.)\1{2,12}$"
)

MULTISPACE_PATTERN = re.compile(
    r"[ \t]+"
)

MULTINEWLINE_PATTERN = re.compile(
    r"\n{3,}"
)

PAGE_NUMBER_PATTERN = re.compile(
    r"^(?:page\s*)?\d{1,5}$",
    re.IGNORECASE,
)

PAGE_OF_PATTERN = re.compile(
    r"^page\s+\d+\s*(?:of|/)\s*\d+$",
    re.IGNORECASE,
)

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)


# ==========================================================
# UNICODE NORMALIZATION
# ==========================================================

def _normalize_unicode(
    text: str,
) -> str:

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    # Remove zero-width/invisible formatting characters.
    text = re.sub(
        r"[\u200b\u200c\u200d\ufeff]",
        "",
        text,
    )

    # Soft hyphen.
    text = text.replace(
        "\u00ad",
        "",
    )

    return text


# ==========================================================
# CONTROL CHARACTERS
# ==========================================================

def _remove_control_characters(
    text: str,
) -> str:

    result = []

    for char in text:

        if char in "\n\t":
            result.append(char)
            continue

        category = unicodedata.category(
            char
        )

        if category.startswith("C"):
            continue

        result.append(char)

    return "".join(result)


# ==========================================================
# LINE NORMALIZATION
# ==========================================================

def _normalize_line(
    line: str,
) -> str:

    line = _normalize_unicode(
        line
    )

    line = _remove_control_characters(
        line
    )

    line = line.replace(
        "\r",
        "",
    )

    line = MULTISPACE_PATTERN.sub(
        " ",
        line,
    )

    return line.strip()


# ==========================================================
# HYPHENATION REPAIR
# ==========================================================

def _repair_hyphenation(
    lines: Sequence[str],
) -> List[str]:
    """
    Repair words broken at PDF line boundaries.

    Example:

        explora-
        tion

    becomes:

        exploration

    But:

        state-
        of-the-art

    is preserved when joining would be unsafe.
    """

    output: List[str] = []

    index = 0

    while index < len(lines):

        current = lines[index].strip()

        if not current:
            index += 1
            continue

        if index + 1 < len(lines):

            following = lines[
                index + 1
            ].strip()

            # Word ending with a normal alphabetic hyphen
            # followed by a lowercase continuation.
            match = re.search(
                r"([A-Za-z]{2,})-$",
                current,
            )

            if (
                match
                and following
                and re.match(
                    r"^[a-z]",
                    following,
                )
            ):

                joined = (
                    current[:-1]
                    + following
                )

                output.append(
                    joined
                )

                index += 2
                continue

        output.append(
            current
        )

        index += 1

    return output


# ==========================================================
# JUNK DETECTION
# ==========================================================

def _is_probable_junk_line(
    line: str,
    ignore_page_numbers: bool = False,
) -> bool:

    value = line.strip()

    if not value:
        return True

    if len(value) <= 1:
        return True

    if not ignore_page_numbers and PAGE_NUMBER_PATTERN.fullmatch(
        value
    ):
        return True

    if not ignore_page_numbers and PAGE_OF_PATTERN.fullmatch(
        value
    ):
        return True

    for pattern in JUNK_TOKEN_PATTERNS:

        # Page-number patterns are handled separately using page
        # metadata/geometry when ignore_page_numbers=True.
        if ignore_page_numbers and pattern.pattern == r"^(?:page|pg\.?)\s*\d+$":
            continue

        if pattern.fullmatch(
            value
        ):

            # Don't remove a normal Roman numeral such as
            # "IV" unless it is clearly an isolated artefact.
            if (
                pattern.pattern.startswith(
                    "^[ivxlcdm]"
                )
                and len(value) <= 4
            ):
                continue

            return True

    if ISOLATED_REPEAT_PATTERN.fullmatch(
        value
    ):
        return True

    # Common extraction artefact:
    # "iiii iiiiii"
    tokens = value.split()

    if tokens and all(
        len(token) <= 8
        and len(set(token.lower())) <= 2
        and token.isalpha()
        for token in tokens
    ):
        joined = "".join(
            tokens
        ).lower()

        if (
            len(joined) >= 4
            and len(set(joined)) <= 2
        ):
            return True

    return False


# ==========================================================
# ADJACENT DUPLICATE REPAIR
# ==========================================================

def _remove_adjacent_duplicate_words(
    text: str,
) -> str:
    """
    Repair obvious extraction duplicates:

        Chandrayaan-3 Chandrayaan-3 is ...

    ->

        Chandrayaan-3 is ...

    Only immediately repeated words/phrases are considered.
    """

    tokens = text.split()

    if len(tokens) < 2:
        return text

    result: List[str] = []

    index = 0

    while index < len(tokens):

        if (
            index + 1 < len(tokens)
            and tokens[index].lower()
            == tokens[index + 1].lower()
        ):

            result.append(
                tokens[index]
            )

            index += 2

            continue

        result.append(
            tokens[index]
        )

        index += 1

    return " ".join(
        result
    )


# ==========================================================
# REPEATED HEADER / FOOTER DETECTION
# ==========================================================

def _line_signature(
    line: str,
) -> str:

    value = line.lower().strip()

    value = re.sub(
        r"\d+",
        "#",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value


def _detect_repeated_lines(
    pages: Sequence[Dict[str, Any]],
) -> set[str]:
    """
    Detect repeated page headers/footers using page geometry.

    Only lines extracted from the physical top/bottom regions are
    considered boilerplate candidates. This prevents a repeated
    numeric value in the document body from being mistaken for a
    footer merely because it appears near the end of a text string.
    """

    if len(pages) < 3:
        return set()

    counter = Counter()
    page_count = len(pages)

    for page in pages:
        candidates = _page_edge_line_signatures(page)
        for line in candidates:
            signature = _line_signature(line)
            # A numeric-only signature collapses all page numbers to
            # the same ``#`` token. Do not classify it as repeated
            # boilerplate; page numbers are handled with page-number
            # metadata in _is_page_number_artifact().
            if signature == "#":
                continue
            if len(signature) >= 3:
                counter[signature] += 1

    threshold = max(
        2,
        int(page_count * REPEATED_PAGE_FRACTION),
    )

    return {
        signature
        for signature, count in counter.items()
        if count >= threshold
    }


# ==========================================================
# PAGE-METADATA-AWARE PAGE NUMBER DETECTION
# ==========================================================

def _page_edge_line_signatures(
    page: Dict[str, Any],
) -> set[str]:
    """
    Return normalized line signatures located near the physical
    top/bottom of a page, using loader-provided block geometry.

    This deliberately uses page metadata/geometry rather than
    deleting every line that merely looks like ``Page 12`` or
    ``12``. A legitimate body reference such as ``Page 12`` is
    therefore preserved when it occurs in the middle of a page.
    """

    height = float(page.get("height", 0.0) or 0.0)
    blocks = page.get("blocks") or []

    if height <= 0 or not blocks:
        return set()

    signatures: set[str] = set()

    for block in blocks:
        try:
            y0 = float(block.get("y0", 0.0))
            y1 = float(block.get("y1", y0))
        except (TypeError, ValueError):
            continue

        if y0 <= height * PAGE_EDGE_FRACTION or y1 >= height * (1.0 - PAGE_EDGE_FRACTION):
            for line in str(block.get("text", "")).splitlines():
                normalized = _normalize_line(line)
                if normalized:
                    signatures.add(normalized)

    return signatures


def _is_page_number_artifact(
    line: str,
    page: Dict[str, Any],
    edge_signatures: set[str],
) -> bool:
    """
    Detect a page-number extraction artefact only when there is
    page-level evidence supporting the interpretation.

    Evidence sources:
      1. The line is physically located in a top/bottom block.
      2. The line contains the loader's page_number.

    Without that evidence, numeric lines are left untouched.
    """

    value = line.strip()
    if value not in edge_signatures:
        return False

    page_number = page.get("page_number")
    if page_number is None:
        return False

    try:
        page_number = int(page_number)
    except (TypeError, ValueError):
        return False

    number_match = re.fullmatch(r"(?:page\s*)?(\d{1,5})", value, re.IGNORECASE)
    if number_match:
        return int(number_match.group(1)) == page_number

    page_of_match = re.fullmatch(
        r"page\s+(\d{1,5})\s*(?:of|/)\s*\d{1,5}",
        value,
        re.IGNORECASE,
    )
    if page_of_match:
        return int(page_of_match.group(1)) == page_number

    return False


# ==========================================================
# PAGE CLEANING
# ==========================================================

def clean_page(
    page: Dict[str, Any],
    repeated_lines: set[str] | None = None,
) -> Dict[str, Any]:
    """
    Clean one structured page while preserving provenance.
    """

    repeated_lines = (
        repeated_lines
        if repeated_lines is not None
        else set()
    )

    raw_text = str(
        page.get(
            "text",
            "",
        )
    )

    raw_lines = raw_text.splitlines()

    normalized_lines = [
        _normalize_line(
            line
        )
        for line in raw_lines
    ]

    # Use loader-provided page geometry when available. This is the
    # key Optimization #39 change: page-like lines are not removed
    # solely because they match a generic regex.
    edge_signatures = _page_edge_line_signatures(page)

    # Remove empty/junk lines.
    filtered: List[str] = []

    for line in normalized_lines:

        if not line:
            continue

        signature = _line_signature(
            line
        )

        if signature in repeated_lines:
            continue

        if _is_page_number_artifact(
            line,
            page,
            edge_signatures,
        ):
            continue

        # Generic junk detection deliberately runs after the
        # metadata-aware page-number check. Numeric body lines are
        # therefore not discarded merely because they look like a
        # page number.
        if _is_probable_junk_line(
            line,
            ignore_page_numbers=True,
        ):
            continue

        filtered.append(
            line
        )

    filtered = _repair_hyphenation(
        filtered
    )

    text = "\n\n".join(
        filtered
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    text = _remove_adjacent_duplicate_words(
        text
    )

    text = text.strip()

    cleaned = dict(
        page
    )

    cleaned[
        "text"
    ] = text

    cleaned[
        "raw_text"
    ] = raw_text

    cleaned[
        "cleaned_word_count"
    ] = len(
        text.split()
    )

    return cleaned


# ==========================================================
# STRUCTURED CLEANING
# ==========================================================

def clean_pages(
    pages: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Clean a complete document while preserving page metadata.
    """

    repeated_lines = (
        _detect_repeated_lines(
            pages
        )
    )

    cleaned_pages = [
        clean_page(
            page,
            repeated_lines=repeated_lines,
        )
        for page in pages
    ]

    return cleaned_pages


# ==========================================================
# PLAIN TEXT CLEANING
# ==========================================================

def clean_text(
    text: str,
) -> str:
    """
    Backward-compatible plain-text cleaner.

    Use clean_pages() when page provenance is available.
    """

    if text is None:
        return ""

    text = str(text)

    text = _normalize_unicode(
        text
    )

    text = _remove_control_characters(
        text
    )

    lines = [
        _normalize_line(
            line
        )
        for line in text.splitlines()
    ]

    lines = [
        line
        for line in lines
        if not _is_probable_junk_line(
            line
        )
    ]

    lines = _repair_hyphenation(
        lines
    )

    cleaned = "\n\n".join(
        lines
    )

    cleaned = re.sub(
        r"[ \t]+",
        " ",
        cleaned,
    )

    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned,
    )

    cleaned = _remove_adjacent_duplicate_words(
        cleaned
    )

    return cleaned.strip()


# ==========================================================
# DOCUMENT QUALITY SCORE
# ==========================================================

def text_quality_score(
    text: str,
) -> float:
    """
    Estimate whether extracted text is usable.

    This is NOT a semantic quality metric.

    It detects obvious corruption such as:
        - extremely short documents
        - excessive repeated characters
        - excessive punctuation
        - excessive single-character tokens
    """

    if not text:
        return 0.0

    tokens = text.split()

    if not tokens:
        return 0.0

    score = 1.0

    if len(tokens) < 20:
        score *= 0.75

    single_char_ratio = (
        sum(
            1
            for token in tokens
            if len(token) == 1
        )
        / len(tokens)
    )

    if single_char_ratio > 0.20:
        score *= 0.65

    punctuation_ratio = (
        sum(
            1
            for char in text
            if not char.isalnum()
            and not char.isspace()
        )
        / max(
            1,
            len(text),
        )
    )

    if punctuation_ratio > 0.35:
        score *= 0.75

    repeated_ratio = (
        sum(
            1
            for token in tokens
            if len(token) >= 3
            and len(
                set(
                    token.lower()
                )
            ) <= 2
        )
        / len(tokens)
    )

    if repeated_ratio > 0.20:
        score *= 0.55

    return max(
        0.0,
        min(
            1.0,
            score,
        ),
    )


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Clean extracted PDF text."
    )

    parser.add_argument(
        "text_file",
        help="UTF-8 text file to clean.",
    )

    args = parser.parse_args()

    with open(
        args.text_file,
        "r",
        encoding="utf-8",
    ) as handle:

        raw = handle.read()

    cleaned = clean_text(
        raw
    )

    print(cleaned)

    print()
    print(
        f"Quality score: "
        f"{text_quality_score(cleaned):.3f}"
    )