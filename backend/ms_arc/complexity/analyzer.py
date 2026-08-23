import re
from typing import List

import spacy

from backend.ms_arc.state.retrieval_state import RetrievalState
from backend.ms_arc.utils.helpers import normalize

from backend.models.model_registry import ModelRegistry

# Load spaCy model once
nlp = ModelRegistry.get_nlp()


class QueryComplexityAnalyzer:
    """
    Analyze the complexity of a user query.

    Signals:
    1. Query Length
    2. Named Entities
    3. Technical Keyword Density
    4. Multi-hop Reasoning
    """

    def __init__(self):

        self.technical_keywords = {

            "algorithm",
            "database",
            "embedding",
            "retrieval",
            "vector",
            "faiss",
            "bm25",
            "transformer",
            "neural",
            "rag",
            "llm",
            "token",
            "ranking",
            "crossencoder",
            "pipeline",
            "api",
            "optimization",
            "graph",
            "reasoning",
            "memory",
            "semantic",
            "index",
            "encoder",
            "decoder",
            "precision",
            "recall",
            "latency",
            "similarity"

        }

        self.multihop_words = {

            "compare",
            "difference",
            "between",
            "relation",
            "why",
            "how",
            "explain",
            "connect",
            "versus",
            "vs",
            "impact"

        }

    def analyze(self, state: RetrievalState) -> RetrievalState:

        query = state.query.strip()

        doc = nlp(query)

        tokens = [
            token.text.lower()
            for token in doc
            if not token.is_punct
        ]

        # --------------------------
        # Feature 1 : Query Length
        # --------------------------

        query_length = len(tokens)

        length_score = normalize(query_length, 2, 30)

        # --------------------------
        # Feature 2 : Named Entities
        # --------------------------

        entity_count = len(doc.ents)

        entity_score = normalize(entity_count, 0, 6)

        # --------------------------
        # Feature 3 : Technical Density
        # --------------------------

        technical_count = sum(
            token in self.technical_keywords
            for token in tokens
        )

        density = 0

        if len(tokens):

            density = technical_count / len(tokens)

        technical_score = min(density * 3.0, 1.0)

        # --------------------------
        # Feature 4 : Multi-hop
        # --------------------------

        multihop = any(

            word in tokens

            for word in self.multihop_words

        )

        multihop_score = 1.0 if multihop else 0.0

        # --------------------------
        # Final Complexity
        # --------------------------

        complexity = (

            0.30 * length_score +
            0.20 * entity_score +
            0.25 * technical_score +
            0.25 * multihop_score

        )

        state.query_complexity = round(complexity, 3)

        # --------------------------
        # Query Type
        # --------------------------

        if complexity < 0.30:

            query_type = "simple"

            topk = 3

        elif complexity < 0.55:

            query_type = "medium"

            topk = 5

        elif complexity < 0.75:

            query_type = "complex"

            topk = 8

        else:

            query_type = "very_complex"

            topk = 10

        state.query_type = query_type

        state.recommended_topk = topk

        state.debug["complexity"] = {

            "query_length": query_length,

            "named_entities": entity_count,

            "technical_keywords": technical_count,

            "technical_density": round(density, 3),

            "multihop": multihop,

            "length_score": round(length_score, 3),

            "entity_score": round(entity_score, 3),

            "technical_score": round(technical_score, 3),

            "multihop_score": multihop_score

        }

        return state