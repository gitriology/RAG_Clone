"""Lightweight query-complexity analysis for MS-ARC.

Optimization #24/#25
--------------------
The complexity signal remains part of MS-ARC's research contribution, but
query analysis no longer requires loading/running spaCy for every query.

Modes:
- ``lightweight`` (default): regex/token based features; no spaCy startup cost.
- ``spacy``: preserves the original NER-backed entity feature when explicitly
  requested with ``MSARC_COMPLEXITY_NER=spacy``.

Both modes preserve the four research features:
query length, entity signal, technical density, and multi-hop language.
"""

from __future__ import annotations

import os
import re
from typing import List

from backend.ms_arc.state.retrieval_state import RetrievalState
from backend.ms_arc.utils.helpers import normalize
from backend.retrieval.query_matching import analyze_query


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
PROPER_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z0-9-]{1,}\b")
ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")


class QueryComplexityAnalyzer:
    """Analyze query complexity with dependency-light defaults."""

    def __init__(self, ner_mode: str | None = None):
        self.technical_keywords = {
            "algorithm", "database", "embedding", "retrieval", "vector",
            "faiss", "bm25", "transformer", "neural", "rag", "llm",
            "token", "ranking", "crossencoder", "pipeline", "api",
            "optimization", "graph", "reasoning", "memory", "semantic",
            "index", "encoder", "decoder", "precision", "recall",
            "latency", "similarity",
        }
        self.multihop_words = {
            "compare", "difference", "between", "relation", "why", "how",
            "explain", "connect", "versus", "vs", "impact",
        }

        configured = ner_mode or os.getenv("MSARC_COMPLEXITY_NER", "lightweight")
        self.ner_mode = str(configured).strip().lower()
        if self.ner_mode not in {"lightweight", "spacy"}:
            self.ner_mode = "lightweight"
        self._nlp = None

    def _get_nlp(self):
        """Lazy-load spaCy only when the experiment explicitly requests it."""
        if self._nlp is None:
            from backend.models.model_registry import ModelRegistry
            self._nlp = ModelRegistry.get_nlp()
        return self._nlp

    @staticmethod
    def _lightweight_entity_count(query: str, tokens: List[str]) -> int:
        """Cheap proxy for entity signal without a full NER pipeline.

        Acronyms and capitalized multi-character tokens are counted as entity
        candidates. The count is capped to the same [0, 6] normalization range
        used by the original spaCy implementation.
        """
        acronyms = len(ACRONYM_RE.findall(query))
        proper_tokens = sum(
            1 for token in tokens
            if PROPER_TOKEN_RE.fullmatch(token)
            and not token.lower() in {"what", "who", "where", "when", "why", "how"}
        )
        return min(6, max(acronyms, proper_tokens))

    def _tokenize(self, query: str) -> List[str]:
        return [m.group(0).lower() for m in TOKEN_RE.finditer(query)]

    def analyze(self, state: RetrievalState) -> RetrievalState:
        query = state.query.strip()
        query_analysis = analyze_query(query)
        tokens = self._tokenize(query)

        # --------------------------------------------------
        # Feature 1: Query Length
        # --------------------------------------------------
        query_length = len(tokens)
        length_score = normalize(query_length, 2, 30)

        # --------------------------------------------------
        # Feature 2: Named/entity signal
        # --------------------------------------------------
        if self.ner_mode == "spacy":
            doc = self._get_nlp()(query)
            entity_count = len(doc.ents)
        else:
            entity_count = self._lightweight_entity_count(query, tokens)
        entity_score = normalize(entity_count, 0, 6)

        # --------------------------------------------------
        # Feature 3: Technical Density
        # --------------------------------------------------
        technical_count = sum(token in self.technical_keywords for token in tokens)
        density = technical_count / len(tokens) if tokens else 0.0
        technical_score = min(density * 3.0, 1.0)

        # --------------------------------------------------
        # Feature 4: Multi-hop
        # --------------------------------------------------
        multihop = any(word in tokens for word in self.multihop_words)
        multihop_score = 1.0 if multihop else 0.0

        # --------------------------------------------------
        # Final Complexity — preserved for #25
        # --------------------------------------------------
        complexity = (
            0.30 * length_score
            + 0.20 * entity_score
            + 0.25 * technical_score
            + 0.25 * multihop_score
        )
        state.query_complexity = round(complexity, 3)

        if complexity < 0.30:
            query_type, topk = "simple", 3
        elif complexity < 0.55:
            query_type, topk = "medium", 5
        elif complexity < 0.75:
            query_type, topk = "complex", 8
        else:
            query_type, topk = "very_complex", 10

        intents = set(query_analysis.get("intents", []))
        if "compare" in intents:
            topk = max(topk, 10)
        elif "why" in intents or "how" in intents or "explain" in intents:
            topk = max(topk, 8)
        elif len(query_analysis.get("focus_phrases", [])) >= 2:
            topk = max(topk, 6)

        state.query_type = query_type
        state.recommended_topk = topk
        state.debug["query_analysis"] = query_analysis
        state.debug["complexity"] = {
            "query_length": query_length,
            "named_entities": entity_count,
            "entity_signal_mode": self.ner_mode,
            "technical_keywords": technical_count,
            "technical_density": round(density, 3),
            "multihop": multihop,
            "length_score": round(length_score, 3),
            "entity_score": round(entity_score, 3),
            "technical_score": round(technical_score, 3),
            "multihop_score": multihop_score,
        }
        return state
