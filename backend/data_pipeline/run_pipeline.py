"""
Unified production data preprocessing pipeline.

Pipeline
--------
PDF
 ↓
layout-aware extraction
 ↓
page-aware cleaning
 ↓
header/footer detection
 ↓
hyphenation repair
 ↓
junk removal
 ↓
semantic sentence segmentation
 ↓
semantic chunking
 ↓
metadata/provenance
 ↓
all_domains.json

The pipeline automatically discovers:

    data/raw/<domain>/*.pdf

Examples:

    data/raw/healthcare/*.pdf
    data/raw/space_missions/*.pdf
    data/raw/legal/*.pdf
    data/raw/education/*.pdf

No domain should need a separate preprocessing implementation.

Important
---------
After rebuilding the dataset, embeddings/FAISS/BM25 must also be
rebuilt because their document ordering/content depends on this
dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


# ==========================================================
# IMPORTS
# ==========================================================

from backend.data_pipeline.ingestion.load_pdf import (
    load_pdf,
)

from backend.data_pipeline.cleaning.clean_text import (
    clean_pages,
    text_quality_score,
)

from backend.data_pipeline.chunking.chunker import (
    chunk_pages,
    chunk_quality_score,
)


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ==========================================================
# DEFAULT PATHS
# ==========================================================

DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

DEFAULT_OUTPUT_FILE = (
    DEFAULT_OUTPUT_DIR
    / "all_domains.json"
)

DEFAULT_MANIFEST = (
    DEFAULT_OUTPUT_DIR
    / "preprocessing_manifest.json"
)


# ==========================================================
# CHUNK CONFIGURATION
# ==========================================================

CHUNK_SIZE = 300
CHUNK_OVERLAP = 45

MIN_DOCUMENT_WORDS = 10


# ==========================================================
# PIPELINE VERSION
# ==========================================================

PREPROCESSING_VERSION = (
    "unified-pdf-pipeline-v2"
)


# ==========================================================
# SAFE NAME
# ==========================================================

def safe_name(
    value: str,
) -> str:
    """
    Convert an arbitrary directory/file name into a stable
    JSON-safe identifier.
    """

    value = (
        str(value)
        .strip()
        .lower()
    )

    value = re.sub(
        r"[^a-z0-9._-]+",
        "_",
        value,
    )

    value = re.sub(
        r"_+",
        "_",
        value,
    )

    return (
        value.strip("_")
        or "general"
    )


# ==========================================================
# DOMAIN INFERENCE
# ==========================================================

def infer_domain(
    pdf_path: Path,
    input_root: Path,
) -> str:
    """
    Infer domain from the first directory beneath data/raw.

    Example:

        data/raw/space_missions/foo.pdf
                     ^^^^^^^^^^^^^
                     domain
    """

    try:

        relative = (
            pdf_path
            .resolve()
            .relative_to(
                input_root.resolve()
            )
        )

    except ValueError:

        return "general"

    parts = relative.parts

    if len(parts) >= 2:
        return safe_name(
            parts[0]
        )

    return "general"


# ==========================================================
# FILE HASH
# ==========================================================

def file_sha256(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        while True:

            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


# ==========================================================
# TEXT FINGERPRINT
# ==========================================================

def text_sha256(
    text: str,
) -> str:

    return hashlib.sha256(
        str(text)
        .encode(
            "utf-8",
            errors="replace",
        )
    ).hexdigest()


# ==========================================================
# PDF DISCOVERY
# ==========================================================

def discover_pdfs(
    input_root: Path,
) -> List[Path]:

    if not input_root.exists():

        raise FileNotFoundError(
            f"Input directory does not exist: "
            f"{input_root}"
        )

    pdfs = [
        path
        for path in input_root.rglob(
            "*.pdf"
        )
        if path.is_file()
    ]

    # Include uppercase extensions.
    pdfs.extend(
        path
        for path in input_root.rglob(
            "*.PDF"
        )
        if path.is_file()
    )

    # Deduplicate and sort deterministically.
    unique = sorted(
        {
            path.resolve()
            for path in pdfs
        },
        key=lambda path:
        str(path).lower(),
    )

    return unique


# ==========================================================
# SOURCE TYPE
# ==========================================================

def infer_document_type(
    domain: str,
    source_name: str,
) -> str:

    domain_lower = (
        domain.lower()
    )

    if "health" in domain_lower:
        return "medical_guideline"

    if "legal" in domain_lower:
        return "legal_document"

    if "education" in domain_lower:
        return "educational_document"

    if (
        "space" in domain_lower
        or "science" in domain_lower
    ):
        return "scientific_document"

    return "document"


# ==========================================================
# CHUNK METADATA
# ==========================================================

def build_chunk_record(
    chunk: Dict[str, Any],
    *,
    document_id: str,
    numeric_id: int,
    source: str,
    source_path: str,
    domain: str,
    document_type: str,
    source_file_hash: str,
) -> Dict[str, Any]:

    text = str(
        chunk.get(
            "text",
            "",
        )
    ).strip()

    page_start = chunk.get(
        "page_start"
    )

    page_end = chunk.get(
        "page_end"
    )

    return {
        # --------------------------------------------------
        # Existing schema compatibility
        # --------------------------------------------------

        "id": numeric_id,

        "text": text,

        "domain": domain,

        "source": source,

        "type": document_type,

        # --------------------------------------------------
        # Stable provenance
        # --------------------------------------------------

        "document_id": document_id,

        "source_path": source_path,

        "source_file_hash": source_file_hash,

        "chunk_index": int(
            chunk.get(
                "chunk_index",
                0,
            )
        ),

        "page_start": (
            int(page_start)
            if page_start is not None
            else None
        ),

        "page_end": (
            int(page_end)
            if page_end is not None
            else None
        ),

        # --------------------------------------------------
        # Quality metadata
        # --------------------------------------------------

        "word_count": int(
            chunk.get(
                "word_count",
                len(
                    text.split()
                ),
            )
        ),

        "chunk_quality": round(
            chunk_quality_score(
                text
            ),
            4,
        ),

        "text_hash": text_sha256(
            text
        ),

        # --------------------------------------------------
        # Versioning
        # --------------------------------------------------

        "preprocessing_version":
            PREPROCESSING_VERSION,
    }


# ==========================================================
# PROCESS ONE PDF
# ==========================================================

def process_pdf(
    pdf_path: Path,
    input_root: Path,
    starting_id: int,
) -> Dict[str, Any]:
    """
    Process one PDF completely.
    """

    domain = infer_domain(
        pdf_path,
        input_root,
    )

    source = pdf_path.name

    document_id = hashlib.sha256(
        str(
            pdf_path
            .resolve()
        )
        .encode(
            "utf-8"
        )
    ).hexdigest()[:16]

    source_hash = file_sha256(
        pdf_path
    )

    print()
    print(
        "=" * 70
    )

    print(
        f"[PDF] Processing: {source}"
    )

    print(
        f"[PDF] Domain: {domain}"
    )

    print(
        f"[PDF] Document ID: {document_id}"
    )

    # ------------------------------------------------------
    # 1. LAYOUT-AWARE EXTRACTION
    # ------------------------------------------------------

    pages = load_pdf(
        pdf_path,
        return_pages=True,
    )

    print(
        f"[PDF] Pages extracted: "
        f"{len(pages)}"
    )

    raw_words = sum(
        len(
            str(
                page.get(
                    "text",
                    "",
                )
            ).split()
        )
        for page in pages
    )

    # ------------------------------------------------------
    # 2. CLEANING
    # ------------------------------------------------------

    cleaned_pages = clean_pages(
        pages
    )

    cleaned_words = sum(
        int(
            page.get(
                "cleaned_word_count",
                len(
                    str(
                        page.get(
                            "text",
                            "",
                        )
                    ).split()
                ),
            )
        )
        for page in cleaned_pages
    )

    print(
        f"[Clean] Raw words: "
        f"{raw_words}"
    )

    print(
        f"[Clean] Clean words: "
        f"{cleaned_words}"
    )

    # ------------------------------------------------------
    # 3. DOCUMENT QUALITY
    # ------------------------------------------------------

    full_cleaned_text = "\n\n".join(
        str(
            page.get(
                "text",
                "",
            )
        )
        for page in cleaned_pages
        if page.get("text")
    ).strip()

    document_quality = (
        text_quality_score(
            full_cleaned_text
        )
    )

    print(
        f"[Clean] Quality: "
        f"{document_quality:.3f}"
    )

    # Do not index completely empty/corrupt PDFs.
    if (
        len(
            full_cleaned_text.split()
        )
        < MIN_DOCUMENT_WORDS
    ):

        print(
            "[PDF] SKIPPED: insufficient "
            "usable text."
        )

        return {
            "records": [],
            "document": {
                "document_id": document_id,
                "source": source,
                "domain": domain,
                "pages": len(pages),
                "raw_words": raw_words,
                "cleaned_words": cleaned_words,
                "document_quality": round(
                    document_quality,
                    4,
                ),
                "chunks": 0,
                "skipped": True,
                "reason":
                    "insufficient_usable_text",
            },
        }

    # ------------------------------------------------------
    # 4. SEMANTIC CHUNKING
    # ------------------------------------------------------

    chunks = chunk_pages(
        cleaned_pages,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    print(
        f"[Chunk] Chunks created: "
        f"{len(chunks)}"
    )

    # ------------------------------------------------------
    # 5. METADATA
    # ------------------------------------------------------

    document_type = (
        infer_document_type(
            domain,
            source,
        )
    )

    records: List[
        Dict[str, Any]
    ] = []

    current_id = starting_id

    for chunk in chunks:

        text = str(
            chunk.get(
                "text",
                "",
            )
        ).strip()

        if not text:
            continue

        quality = chunk_quality_score(
            text
        )

        # Do not index severely corrupted chunks.
        if quality < 0.30:
            print(
                "[Chunk] Skipped low-quality "
                f"chunk {chunk.get('chunk_index')}: "
                f"quality={quality:.3f}"
            )
            continue

        record = build_chunk_record(
            chunk,
            document_id=document_id,
            numeric_id=current_id,
            source=source,
            source_path=str(
                pdf_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            domain=domain,
            document_type=document_type,
            source_file_hash=source_hash,
        )

        records.append(
            record
        )

        current_id += 1

    print(
        f"[Chunk] Indexed chunks: "
        f"{len(records)}"
    )

    print(
        "=" * 70
    )

    return {
        "records": records,
        "document": {
            "document_id": document_id,
            "source": source,
            "source_path": str(
                pdf_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "source_file_hash": source_hash,
            "domain": domain,
            "type": document_type,
            "pages": len(pages),
            "raw_words": raw_words,
            "cleaned_words": cleaned_words,
            "document_quality": round(
                document_quality,
                4,
            ),
            "chunks": len(records),
            "skipped": False,
        },
    }


# ==========================================================
# ATOMIC JSON SAVE
# ==========================================================

def atomic_json_save(
    data: Any,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}_",
            suffix=".tmp",
            delete=False,
        ) as handle:

            temporary_path = Path(
                handle.name
            )

            json.dump(
                data,
                handle,
                ensure_ascii=False,
                indent=2,
            )

            handle.flush()
            os.fsync(
                handle.fileno()
            )

        os.replace(
            temporary_path,
            path,
        )

    finally:

        if (
            temporary_path is not None
            and temporary_path.exists()
        ):

            try:
                temporary_path.unlink()
            except OSError:
                pass


# ==========================================================
# DOMAIN OUTPUT
# ==========================================================

def save_domain_files(
    records: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> None:

    domains = sorted(
        {
            record.get(
                "domain",
                "general",
            )
            for record in records
        }
    )

    for domain in domains:

        domain_records = [
            record
            for record in records
            if record.get(
                "domain"
            ) == domain
        ]

        clean_path = (
            output_dir
            / f"{domain}_clean.json"
        )

        chunks_path = (
            output_dir
            / f"{domain}_chunks.json"
        )

        atomic_json_save(
            domain_records,
            clean_path,
        )

        atomic_json_save(
            domain_records,
            chunks_path,
        )


# ==========================================================
# MANIFEST
# ==========================================================

def build_manifest(
    records: Sequence[Dict[str, Any]],
    documents: Sequence[Dict[str, Any]],
    input_dir: Path,
    output_file: Path,
) -> Dict[str, Any]:

    domain_counts: Dict[
        str,
        int
    ] = {}

    for record in records:

        domain = record.get(
            "domain",
            "general",
        )

        domain_counts[
            domain
        ] = (
            domain_counts.get(
                domain,
                0,
            )
            + 1
        )

    return {
        "preprocessing_version":
            PREPROCESSING_VERSION,

        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "input_dir":
            str(input_dir),

        "output_file":
            str(output_file),

        "chunk_size":
            CHUNK_SIZE,

        "chunk_overlap":
            CHUNK_OVERLAP,

        "document_count":
            len(documents),

        "indexed_document_count":
            sum(
                1
                for document in documents
                if not document.get(
                    "skipped",
                    False,
                )
            ),

        "chunk_count":
            len(records),

        "domain_counts":
            domain_counts,

        "documents":
            list(documents),
    }


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def run_pipeline(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_file: Path = DEFAULT_OUTPUT_FILE,
) -> List[Dict[str, Any]]:

    input_dir = Path(
        input_dir
    ).resolve()

    output_file = Path(
        output_file
    ).resolve()

    output_dir = (
        output_file.parent
    )

    print()
    print(
        "=" * 70
    )

    print(
        "UNIFIED DATA PREPROCESSING PIPELINE"
    )

    print(
        "=" * 70
    )

    print(
        f"Version: {PREPROCESSING_VERSION}"
    )

    print(
        f"Input:   {input_dir}"
    )

    print(
        f"Output:  {output_file}"
    )

    # ------------------------------------------------------
    # DISCOVER PDFs
    # ------------------------------------------------------

    pdfs = discover_pdfs(
        input_dir
    )

    print()
    print(
        f"[Discovery] PDFs found: "
        f"{len(pdfs)}"
    )

    if not pdfs:

        raise RuntimeError(
            "No PDF files found under "
            f"{input_dir}"
        )

    # ------------------------------------------------------
    # PROCESS
    # ------------------------------------------------------

    all_records: List[
        Dict[str, Any]
    ] = []

    documents: List[
        Dict[str, Any]
    ] = []

    next_id = 0

    failures: List[
        Dict[str, str]
    ] = []

    for pdf_path in pdfs:

        try:

            result = process_pdf(
                pdf_path,
                input_root=input_dir,
                starting_id=next_id,
            )

            records = result[
                "records"
            ]

            document = result[
                "document"
            ]

            all_records.extend(
                records
            )

            documents.append(
                document
            )

            next_id += len(
                records
            )

        except Exception as exc:

            print()
            print(
                f"[ERROR] Failed to process "
                f"{pdf_path}: {exc}"
            )

            failures.append(
                {
                    "file":
                        str(pdf_path),
                    "error":
                        repr(exc),
                }
            )

    # ------------------------------------------------------
    # DETERMINISTIC ID REASSIGNMENT
    # ------------------------------------------------------

    # IDs must correspond exactly to the final JSON ordering.
    for index, record in enumerate(
        all_records
    ):

        record[
            "id"
        ] = index

    # ------------------------------------------------------
    # SAVE MAIN DATASET
    # ------------------------------------------------------

    atomic_json_save(
        all_records,
        output_file,
    )

    # ------------------------------------------------------
    # SAVE DOMAIN DATASETS
    # ------------------------------------------------------

    save_domain_files(
        all_records,
        output_dir,
    )

    # ------------------------------------------------------
    # SAVE MANIFEST
    # ------------------------------------------------------

    manifest = build_manifest(
        records=all_records,
        documents=documents,
        input_dir=input_dir,
        output_file=output_file,
    )

    manifest[
        "failures"
    ] = failures

    atomic_json_save(
        manifest,
        DEFAULT_MANIFEST,
    )

    # ------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "PREPROCESSING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"PDFs discovered: "
        f"{len(pdfs)}"
    )

    print(
        f"Documents processed: "
        f"{len(documents)}"
    )

    print(
        f"Chunks indexed: "
        f"{len(all_records)}"
    )

    print(
        f"Failures: "
        f"{len(failures)}"
    )

    print(
        f"Dataset: "
        f"{output_file}"
    )

    print(
        f"Manifest: "
        f"{DEFAULT_MANIFEST}"
    )

    print()

    domains = sorted(
        {
            record.get(
                "domain",
                "general",
            )
            for record in all_records
        }
    )

    print(
        "Domains:"
    )

    for domain in domains:

        count = sum(
            1
            for record in all_records
            if record.get(
                "domain"
            ) == domain
        )

        print(
            f"  - {domain}: {count}"
        )

    print()

    if failures:

        print(
            "WARNING: Some PDFs failed."
        )

        for failure in failures:

            print(
                f"  {failure['file']}"
            )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The embeddings, FAISS index and BM25 cache "
        "must be rebuilt after this dataset changes."
    )

    print(
        "=" * 70
    )

    return all_records


# ==========================================================
# CLI
# ==========================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Unified layout-aware PDF "
            "preprocessing pipeline."
        )
    )

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT_DIR
        ),
        help=(
            "Root PDF directory. "
            "Default: data/raw"
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_FILE
        ),
        help=(
            "Output dataset. "
            "Default: "
            "data/processed/all_domains.json"
        ),
    )

    args = parser.parse_args()

    run_pipeline(
        input_dir=Path(
            args.input
        ),
        output_file=Path(
            args.output
        ),
    )


if __name__ == "__main__":
    main()