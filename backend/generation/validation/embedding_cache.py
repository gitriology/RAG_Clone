"""Shared MiniLM embedding cache for validation stages.

Optimization #30
-----------------
Context validation and answer validation both use the shared MiniLM model.
This cache avoids encoding the same validation text more than once during a
single request while keeping batch encoding for cache misses.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np


class ValidationEmbeddingCache:
    """Request-local cache of normalized MiniLM embeddings."""

    def __init__(self, model=None):
        self.model = model
        self._cache: Dict[str, np.ndarray] = {}
        self.encode_calls = 0
        self.encoded_items = 0
        self.reused_items = 0

    @staticmethod
    def _key(text: str) -> str:
        return str(text)

    def _ensure_model(self):
        if self.model is None:
            from backend.models.model_registry import ModelRegistry

            self.model = ModelRegistry.get_validation_model()
        return self.model

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """Return normalized embeddings, encoding only cache misses."""
        values = [str(text) for text in texts]
        if not values:
            return np.empty((0, 0), dtype=np.float32)

        model = self._ensure_model()
        missing: List[str] = []
        missing_seen = set()

        for text in values:
            key = self._key(text)
            if key in self._cache:
                self.reused_items += 1
            elif key not in missing_seen:
                missing.append(text)
                missing_seen.add(key)

        if missing:
            encoded = model.encode(
                missing,
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            encoded = np.asarray(encoded, dtype=np.float32)
            if encoded.ndim == 1:
                encoded = encoded.reshape(1, -1)

            self.encode_calls += 1
            self.encoded_items += len(missing)

            for text, vector in zip(missing, encoded):
                self._cache[self._key(text)] = np.asarray(vector, dtype=np.float32)

        return np.vstack([self._cache[self._key(text)] for text in values]).astype(
            np.float32,
            copy=False,
        )

    def info(self) -> Dict[str, int]:
        return {
            "cache_size": len(self._cache),
            "encode_calls": self.encode_calls,
            "encoded_items": self.encoded_items,
            "reused_items": self.reused_items,
        }
