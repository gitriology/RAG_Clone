"""
Production semantic-aware chunker.

Design
------
Prefer:

    document section
        ↓
    paragraph
        ↓
    sentence
        ↓
    token/word budget

instead of blindly slicing every N words.

The public API remains:

    chunk_text(text)

so existing code continues to work.

The production pipeline can additionally use:

    chunk_pages(pages)

to preserve page provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import re


# ==========================================================
# CONFIGURATION
# ==========================================================

DEFAULT_CHUNK_SIZE = 300
DEFAULT_OVERLAP = 45

MIN_CHUNK_WORDS = 45
MAX_CHUNK_WORDS = 360

# Don't create tiny chunks merely because a PDF contains a
# heading/table label.
MIN_SENTENCE_WORDS = 3

# Sentence boundary detection.
SENTENCE_PATTERN = re.compile(
    r"""
    (?<=[.!?])
    (?:
        ["'”’)\]]+
    )?
    \s+
    (?=[A-Z0-9À-ÖØ-Ý])
    """,
    re.VERBOSE,
)

# Preserve common abbreviations.
ABBREVIATIONS = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "fig.",
    "no.",
    "nos.",
    "approx.",
    "jan.",
    "feb.",
    "mar.",
    "apr.",
    "jun.",
    "jul.",
    "aug.",
    "sep.",
    "sept.",
    "oct.",
    "nov.",
    "dec.",
}


# ==========================================================
# DATA TYPE
# ==========================================================

@dataclass
class Chunk:
    text: str
    chunk_index: int
    page_start: int | None = None
    page_end: int | None = None
    word_count: int = 0
    section: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==========================================================
# NORMALIZATION
# ==========================================================

def _normalize_text(
    text: str,
) -> str:

    text = str(
        text or ""
    )

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
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

    return text.strip()


# ==========================================================
# ABBREVIATION PROTECTION
# ==========================================================

def _protect_abbreviations(
    text: str,
) -> Tuple[str, Dict[str, str]]:

    replacements = {}

    for index, abbreviation in enumerate(
        sorted(
            ABBREVIATIONS,
            key=len,
            reverse=True,
        )
    ):

        placeholder = (
            f"__ABBR_{index}__"
        )

        if abbreviation in text.lower():

            pattern = re.compile(
                re.escape(
                    abbreviation
                ),
                re.IGNORECASE,
            )

            text = pattern.sub(
                placeholder,
                text,
            )

            replacements[
                placeholder
            ] = abbreviation

    return text, replacements


def _restore_abbreviations(
    text: str,
    replacements: Dict[str, str],
) -> str:

    for placeholder, value in (
        replacements.items()
    ):

        text = text.replace(
            placeholder,
            value,
        )

    return text


# ==========================================================
# SENTENCE SPLITTING
# ==========================================================

def split_sentences(
    text: str,
) -> List[str]:
    """
    Split text conservatively.

    New paragraphs are always respected.
    """

    text = _normalize_text(
        text
    )

    if not text:
        return []

    paragraphs = re.split(
        r"\n\s*\n",
        text,
    )

    sentences: List[str] = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        protected, replacements = (
            _protect_abbreviations(
                paragraph
            )
        )

        parts = SENTENCE_PATTERN.split(
            protected
        )

        for part in parts:

            part = _restore_abbreviations(
                part.strip(),
                replacements,
            )

            if not part:
                continue

            # A heading/table field should remain intact.
            if (
                len(part.split())
                < MIN_SENTENCE_WORDS
            ):

                if sentences:
                    sentences[-1] += (
                        " " + part
                    )
                else:
                    sentences.append(
                        part
                    )

                continue

            sentences.append(
                part
            )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ==========================================================
# SENTENCE WORD COUNT
# ==========================================================

def _word_count(
    text: str,
) -> int:

    return len(
        text.split()
    )


# ==========================================================
# CHUNK BUILDING
# ==========================================================

def _build_chunks_from_sentences(
    sentences: Sequence[str],
    chunk_size: int,
    overlap: int,
) -> List[str]:

    if not sentences:
        return []

    chunk_size = max(
        MIN_CHUNK_WORDS,
        min(
            MAX_CHUNK_WORDS,
            int(chunk_size),
        ),
    )

    overlap = max(
        0,
        min(
            chunk_size // 2,
            int(overlap),
        ),
    )

    chunks: List[str] = []

    current: List[str] = []
    current_words = 0

    for sentence in sentences:

        sentence_words = _word_count(
            sentence
        )

        # Very long sentence: split only if absolutely
        # necessary.
        if sentence_words > chunk_size:

            if current:
                chunks.append(
                    " ".join(current).strip()
                )

                current = []
                current_words = 0

            words = sentence.split()

            start = 0

            while start < len(words):

                end = min(
                    len(words),
                    start + chunk_size,
                )

                piece = " ".join(
                    words[start:end]
                ).strip()

                if piece:
                    chunks.append(
                        piece
                    )

                if end >= len(words):
                    break

                start = max(
                    start + 1,
                    end - overlap,
                )

            continue

        if (
            current
            and current_words
            + sentence_words
            > chunk_size
        ):

            chunks.append(
                " ".join(
                    current
                ).strip()
            )

            # Sentence-level overlap.
            overlap_words = 0
            overlap_sentences: List[str] = []

            for previous in reversed(
                current
            ):

                previous_words = (
                    _word_count(
                        previous
                    )
                )

                if (
                    overlap_words
                    + previous_words
                    > overlap
                ):
                    break

                overlap_sentences.insert(
                    0,
                    previous,
                )

                overlap_words += (
                    previous_words
                )

            current = overlap_sentences
            current_words = (
                overlap_words
            )

        current.append(
            sentence
        )

        current_words += (
            sentence_words
        )

    if current:
        chunks.append(
            " ".join(
                current
            ).strip()
        )

    return [
        chunk
        for chunk in chunks
        if _word_count(
            chunk
        ) >= MIN_CHUNK_WORDS
    ]


# ==========================================================
# MERGE SMALL TRAILING CHUNKS
# ==========================================================

def _merge_small_chunks(
    chunks: List[str],
) -> List[str]:

    if len(chunks) <= 1:
        return chunks

    result: List[str] = []

    for chunk in chunks:

        words = _word_count(
            chunk
        )

        if (
            result
            and words < MIN_CHUNK_WORDS
        ):

            candidate = (
                result[-1]
                + " "
                + chunk
            )

            if _word_count(
                candidate
            ) <= MAX_CHUNK_WORDS:

                result[-1] = candidate
                continue

        result.append(
            chunk
        )

    return result


# ==========================================================
# PUBLIC TEXT CHUNKER
# ==========================================================

def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[str]:
    """
    Backward-compatible semantic-aware chunking.
    """

    text = _normalize_text(
        text
    )

    if not text:
        return []

    sentences = split_sentences(
        text
    )

    chunks = (
        _build_chunks_from_sentences(
            sentences,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    )

    return _merge_small_chunks(
        chunks
    )


# ==========================================================
# PAGE-AWARE CHUNKING
# ==========================================================

def chunk_pages(
    pages: Sequence[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Chunk cleaned pages while preserving page provenance.

    Page boundaries are respected when possible.

    A chunk is allowed to span pages when the semantic unit
    naturally continues across the page boundary.
    """

    if not pages:
        return []

    document_sentences: List[
        Tuple[str, int]
    ] = []

    for page in pages:

        page_number = int(
            page.get(
                "page_number",
                0,
            )
        )

        text = _normalize_text(
            page.get(
                "text",
                "",
            )
        )

        if not text:
            continue

        for sentence in split_sentences(
            text
        ):

            document_sentences.append(
                (
                    sentence,
                    page_number,
                )
            )

    if not document_sentences:
        return []

    # Build chunks while tracking page boundaries.
    chunks: List[Dict[str, Any]] = []

    current_sentences: List[str] = []
    current_pages: List[int] = []
    current_words = 0

    safe_chunk_size = max(
        MIN_CHUNK_WORDS,
        min(
            MAX_CHUNK_WORDS,
            int(chunk_size),
        ),
    )

    safe_overlap = max(
        0,
        min(
            safe_chunk_size // 2,
            int(overlap),
        ),
    )

    for sentence, page_number in (
        document_sentences
    ):

        sentence_words = _word_count(
            sentence
        )

        # Long sentence.
        if sentence_words > safe_chunk_size:

            if current_sentences:

                chunks.append(
                    {
                        "text":
                            " ".join(
                                current_sentences
                            ).strip(),
                        "page_start":
                            min(
                                current_pages
                            ),
                        "page_end":
                            max(
                                current_pages
                            ),
                    }
                )

                current_sentences = []
                current_pages = []
                current_words = 0

            words = sentence.split()

            start = 0

            while start < len(words):

                end = min(
                    len(words),
                    start + safe_chunk_size,
                )

                piece = " ".join(
                    words[start:end]
                ).strip()

                if (
                    piece
                    and _word_count(
                        piece
                    ) >= MIN_CHUNK_WORDS
                ):

                    chunks.append(
                        {
                            "text": piece,
                            "page_start":
                                page_number,
                            "page_end":
                                page_number,
                        }
                    )

                if end >= len(words):
                    break

                start = max(
                    start + 1,
                    end - safe_overlap,
                )

            continue

        if (
            current_sentences
            and (
                current_words
                + sentence_words
                > safe_chunk_size
            )
        ):

            chunks.append(
                {
                    "text":
                        " ".join(
                            current_sentences
                        ).strip(),
                    "page_start":
                        min(
                            current_pages
                        ),
                    "page_end":
                        max(
                            current_pages
                        ),
                }
            )

            # Sentence overlap.
            overlap_words = 0
            overlap_sentences: List[
                str
            ] = []
            overlap_pages: List[
                int
            ] = []

            for previous_sentence, previous_page in reversed(
                list(
                    zip(
                        current_sentences,
                        current_pages,
                    )
                )
            ):

                previous_words = (
                    _word_count(
                        previous_sentence
                    )
                )

                if (
                    overlap_words
                    + previous_words
                    > safe_overlap
                ):
                    break

                overlap_sentences.insert(
                    0,
                    previous_sentence,
                )

                overlap_pages.insert(
                    0,
                    previous_page,
                )

                overlap_words += (
                    previous_words
                )

            current_sentences = (
                overlap_sentences
            )

            current_pages = (
                overlap_pages
            )

            current_words = (
                overlap_words
            )

        current_sentences.append(
            sentence
        )

        current_pages.append(
            page_number
        )

        current_words += (
            sentence_words
        )

    if current_sentences:

        chunks.append(
            {
                "text":
                    " ".join(
                        current_sentences
                    ).strip(),
                "page_start":
                    min(
                        current_pages
                    ),
                "page_end":
                    max(
                        current_pages
                    ),
            }
        )

    # Remove tiny chunks where possible.
    cleaned_chunks: List[
        Dict[str, Any]
    ] = []

    for chunk in chunks:

        word_count = _word_count(
            chunk["text"]
        )

        if (
            word_count
            < MIN_CHUNK_WORDS
            and cleaned_chunks
        ):

            candidate = (
                cleaned_chunks[-1]["text"]
                + " "
                + chunk["text"]
            )

            if (
                _word_count(candidate)
                <= MAX_CHUNK_WORDS
            ):

                cleaned_chunks[
                    -1
                ]["text"] = candidate

                cleaned_chunks[
                    -1
                ]["page_end"] = chunk[
                    "page_end"
                ]

                continue

        cleaned_chunks.append(
            chunk
        )

    final_chunks = []

    for index, chunk in enumerate(
        cleaned_chunks
    ):

        final_chunks.append(
            {
                "text":
                    chunk["text"].strip(),
                "chunk_index":
                    index,
                "page_start":
                    chunk.get(
                        "page_start"
                    ),
                "page_end":
                    chunk.get(
                        "page_end"
                    ),
                "word_count":
                    _word_count(
                        chunk["text"]
                    ),
            }
        )

    return final_chunks


# ==========================================================
# QUALITY CHECK
# ==========================================================

def chunk_quality_score(
    text: str,
) -> float:

    if not text:
        return 0.0

    words = text.split()

    if len(words) < MIN_CHUNK_WORDS:
        return 0.4

    score = 1.0

    # Excessive one-character tokens usually indicate
    # extraction corruption.
    single_ratio = (
        sum(
            1
            for word in words
            if len(word) == 1
        )
        / len(words)
    )

    if single_ratio > 0.15:
        score *= 0.70

    # Excessive punctuation.
    punctuation = sum(
        1
        for char in text
        if not char.isalnum()
        and not char.isspace()
    )

    punctuation_ratio = (
        punctuation
        / max(1, len(text))
    )

    if punctuation_ratio > 0.30:
        score *= 0.75

    return max(
        0.0,
        min(
            1.0,
            score,
        )
    )


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Semantic-aware text chunker."
    )

    parser.add_argument(
        "text_file",
        help="UTF-8 text file.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
    )

    args = parser.parse_args()

    with open(
        args.text_file,
        "r",
        encoding="utf-8",
    ) as handle:

        text = handle.read()

    chunks = chunk_text(
        text,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    for index, chunk in enumerate(
        chunks
    ):

        print()
        print(
            "=" * 70
        )

        print(
            f"CHUNK {index}"
        )

        print(
            "=" * 70
        )

        print(chunk)

        print(
            f"\nWords: {_word_count(chunk)}"
        )

        print(
            f"Quality: "
            f"{chunk_quality_score(chunk):.3f}"
        )