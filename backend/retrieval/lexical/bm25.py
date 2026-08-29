"""
BM25 lexical retrieval.

Optimization #14
----------------
BM25 is expensive to initialize repeatedly because rank_bm25
constructs its internal statistics from the complete corpus.

This module therefore supports:

    1. Shared text tokenization
    2. Deterministic corpus fingerprinting
    3. Persistent BM25 caching
    4. Automatic cache validation
    5. Automatic rebuild when the corpus changes

The cache contains:
    - BM25Okapi object
    - tokenized corpus
    - corpus fingerprint
    - tokenizer version
    - document count

The cache is only loaded when its metadata matches the
current corpus.

WARNING:
The cache file is a local application artifact and must not
be loaded from an untrusted source because it uses pickle.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from rank_bm25 import BM25Okapi


# ==========================================================
# CONFIGURATION
# ==========================================================

TOKENIZER_VERSION = "bm25-tokenizer-v1"

DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parents[1]
    / "index"
    / "bm25_cache.pkl"
)


# ==========================================================
# TOKENIZATION
# ==========================================================

def tokenize(text: str) -> List[str]:
    """
    Tokenize text for BM25.

    Current policy:

        lowercase
        ->
        whitespace tokenization

    This function is intentionally shared by both document
    indexing and query processing.

    More advanced punctuation handling can be introduced
    later as a separate optimization/ablation without
    changing the cache mechanism.
    """

    if text is None:
        return []

    return str(text).lower().split()


# ==========================================================
# CORPUS PREPARATION
# ==========================================================

def tokenize_corpus(
    texts: Sequence[str],
) -> List[List[str]]:
    """
    Tokenize every document using the shared tokenizer.
    """

    return [
        tokenize(text)
        for text in texts
    ]


# ==========================================================
# CORPUS FINGERPRINT
# ==========================================================

def corpus_fingerprint(
    texts: Sequence[str],
) -> str:
    """
    Create a deterministic fingerprint for the BM25 corpus.

    The fingerprint changes when:

        - document text changes
        - document order changes
        - document count changes

    The tokenizer version is also included so that changing
    tokenization automatically invalidates the old cache.
    """

    hasher = hashlib.sha256()

    hasher.update(
        TOKENIZER_VERSION.encode("utf-8")
    )

    for text in texts:

        normalized = (
            ""
            if text is None
            else str(text)
        )

        encoded = normalized.encode(
            "utf-8",
            errors="replace",
        )

        # Include boundaries and lengths so concatenation
        # cannot accidentally produce the same fingerprint.
        hasher.update(
            len(encoded).to_bytes(
                8,
                byteorder="big",
            )
        )

        hasher.update(encoded)

    return hasher.hexdigest()


# ==========================================================
# BUILD
# ==========================================================

def build_bm25(
    texts: Sequence[str],
) -> Tuple[BM25Okapi, List[List[str]]]:
    """
    Build a BM25 index from scratch.

    This function deliberately performs no caching.

    Use load_or_build_bm25() for the normal application path.
    """

    tokenized = tokenize_corpus(texts)

    bm25 = BM25Okapi(
        tokenized
    )

    return bm25, tokenized


# ==========================================================
# CACHE SAVE
# ==========================================================

def save_bm25_cache(
    bm25: BM25Okapi,
    tokenized: List[List[str]],
    texts: Sequence[str],
    cache_path: str | os.PathLike = DEFAULT_CACHE_PATH,
) -> None:
    """
    Persist BM25 and its tokenized corpus.

    Uses an atomic temporary-file replacement so a partially
    written cache is not normally left behind if the process
    is interrupted during saving.
    """

    cache_path = Path(cache_path)

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "cache_version": 1,
        "tokenizer_version": TOKENIZER_VERSION,
        "corpus_fingerprint": corpus_fingerprint(
            texts
        ),
        "document_count": len(texts),
        "bm25": bm25,
        "tokenized": tokenized,
    }

    temporary_path = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=cache_path.parent,
            prefix=".bm25_cache_",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:

            temporary_path = Path(
                temporary_file.name
            )

            pickle.dump(
                payload,
                temporary_file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

            temporary_file.flush()
            os.fsync(
                temporary_file.fileno()
            )

        os.replace(
            temporary_path,
            cache_path,
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
# CACHE LOAD
# ==========================================================

def load_bm25_cache(
    texts: Sequence[str],
    cache_path: str | os.PathLike = DEFAULT_CACHE_PATH,
):
    """
    Load a BM25 cache if it exists and matches the current
    corpus and tokenizer version.

    Returns:

        (bm25, tokenized)

    or:

        None

    if the cache is missing or invalid.
    """

    cache_path = Path(cache_path)

    if not cache_path.exists():

        return None

    try:

        with cache_path.open(
            "rb"
        ) as cache_file:

            payload = pickle.load(
                cache_file
            )

    except (
        OSError,
        EOFError,
        pickle.PickleError,
        AttributeError,
        ValueError,
        TypeError,
    ) as exc:

        print(
            "[BM25] Cache could not be loaded:"
            f" {exc}"
        )

        return None

    if not isinstance(
        payload,
        dict,
    ):

        print(
            "[BM25] Invalid cache format."
        )

        return None

    if payload.get(
        "cache_version"
    ) != 1:

        print(
            "[BM25] Cache version mismatch."
        )

        return None

    if payload.get(
        "tokenizer_version"
    ) != TOKENIZER_VERSION:

        print(
            "[BM25] Tokenizer version changed."
        )

        return None

    expected_fingerprint = (
        corpus_fingerprint(texts)
    )

    if payload.get(
        "corpus_fingerprint"
    ) != expected_fingerprint:

        print(
            "[BM25] Corpus changed. "
            "Cached BM25 index is stale."
        )

        return None

    if payload.get(
        "document_count"
    ) != len(texts):

        print(
            "[BM25] Document count changed."
        )

        return None

    bm25 = payload.get(
        "bm25"
    )

    tokenized = payload.get(
        "tokenized"
    )

    if bm25 is None or tokenized is None:

        print(
            "[BM25] Cache is missing "
            "required data."
        )

        return None

    if len(tokenized) != len(texts):

        print(
            "[BM25] Tokenized corpus size "
            "does not match dataset."
        )

        return None

    return (
        bm25,
        tokenized,
    )


# ==========================================================
# LOAD OR BUILD
# ==========================================================

def load_or_build_bm25(
    texts: Sequence[str],
    cache_path: str | os.PathLike = DEFAULT_CACHE_PATH,
):
    """
    Load BM25 from cache when possible.

    If the cache does not exist or is stale:

        build BM25
        ->
        save cache
        ->
        return BM25

    Returns
    -------
    bm25:
        BM25Okapi instance

    tokenized:
        Tokenized document corpus

    loaded_from_cache:
        Boolean indicating whether the persistent cache
        was successfully reused.
    """

    cache_path = Path(
        cache_path
    )

    print(
        "[BM25] Checking persistent cache..."
    )

    cached = load_bm25_cache(
        texts,
        cache_path,
    )

    if cached is not None:

        bm25, tokenized = cached

        print(
            "[BM25] Loaded cached BM25 index."
        )

        print(
            f"[BM25] Documents: {len(texts)}"
        )

        print(
            f"[BM25] Cache: {cache_path}"
        )

        return (
            bm25,
            tokenized,
            True,
        )

    print(
        "[BM25] No valid cache found."
    )

    print(
        "[BM25] Building BM25 index..."
    )

    bm25, tokenized = build_bm25(
        texts
    )

    print(
        "[BM25] Saving BM25 cache..."
    )

    try:

        save_bm25_cache(
            bm25=bm25,
            tokenized=tokenized,
            texts=texts,
            cache_path=cache_path,
        )

        print(
            "[BM25] BM25 cache saved."
        )

    except (
        OSError,
        pickle.PickleError,
    ) as exc:

        # Retrieval should still work even if the cache cannot
        # be persisted.
        print(
            "[BM25] WARNING: Could not save "
            f"BM25 cache: {exc}"
        )

    return (
        bm25,
        tokenized,
        False,
    )


# ==========================================================
# SEARCH
# ==========================================================

def search_bm25(
    query: str,
    bm25: BM25Okapi,
    tokenized: Sequence[Sequence[str]],
    k: int = 5,
):
    """
    Search BM25 using the same tokenizer used for documents.
    """

    if k <= 0:

        return []

    query_tokens = tokenize(
        query
    )

    if not query_tokens:

        return []

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    return ranked_indices[:k]