"""
Phase 10.2
Evidence Fusion

Takes the question-aware evidence selected by Phase 10.1
and produces a concise, ordered, evidence-grounded paragraph.

Optimization #17
----------------
Sentence embeddings are produced once by answer_generator.py
and reused here.

No repeated pairwise model.encode() calls.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import util

from backend.models.model_registry import ModelRegistry


# ==========================================================
# OPTIONAL NLP MODEL
# ==========================================================

try:
    nlp = ModelRegistry.get_nlp()
except Exception:
    nlp = None


# ==========================================================
# HELPERS
# ==========================================================

def clean_sentence(text: str) -> str:
    if not text:
        return ""

    text = re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()

    text = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        text,
    )

    return text


def normalize_entity(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text.lower().strip(),
    )


# ==========================================================
# EVIDENCE FUSION
# ==========================================================

class EvidenceFusion:
    """
    Phase 10.2 evidence fusion engine.

    Input
    -----
    generation_result from generate_answer()

    Output
    ------
    {
        "paragraph": "...",
        "graph": [...],
        "groups": [...],
        "entities": {...},
        "sentence_count": ...,
        "embedding_reuses": ...,
        "embedding_fallbacks": ...
    }
    """

    def __init__(
        self,
        sentence_embeddings: Optional[Dict[str, Any]] = None,
    ):
        self.graph = []
        self.entities = defaultdict(list)

        self.sentence_embeddings = (
            sentence_embeddings
            if sentence_embeddings is not None
            else {}
        )

        self.embedding_reuses = 0
        self.embedding_fallbacks = 0

        # Centralized model only used as a fallback.
        self.embedder = (
            ModelRegistry.get_embedding_model()
        )

    # ======================================================
    # PUBLIC ENTRY
    # ======================================================

    def fuse(
        self,
        generation_result: Dict,
    ) -> Dict:

        selected = generation_result.get(
            "selected_sentences",
            [],
        )

        # Prefer embeddings attached to generation_result.
        provided_embeddings = generation_result.get(
            "sentence_embeddings"
        )

        if provided_embeddings:
            self.sentence_embeddings = (
                provided_embeddings
            )

        print()
        print("=" * 70)
        print("Phase 10.2 : Evidence Fusion")
        print("=" * 70)

        print(
            f"Input Sentences : "
            f"{len(selected)}"
        )

        print(
            f"Reusable Embeddings : "
            f"{len(self.sentence_embeddings)}"
        )

        # --------------------------------------------------
        # Empty case
        # --------------------------------------------------

        if not selected:

            print(
                "No selected evidence."
            )

            return {
                "paragraph": "",
                "graph": [],
                "groups": [],
                "entities": {},
                "sentence_count": 0,
                "embedding_reuses": 0,
                "embedding_fallbacks": 0,
            }

        # --------------------------------------------------
        # Step 1
        # --------------------------------------------------

        graph = self._build_sentence_graph(
            selected
        )

        # --------------------------------------------------
        # Step 2
        # --------------------------------------------------

        groups = self._merge_entities(
            graph
        )

        # --------------------------------------------------
        # Step 3
        # --------------------------------------------------

        groups = self._remove_redundancy(
            groups
        )

        # --------------------------------------------------
        # Step 4
        # --------------------------------------------------

        ordered = self._order_sentences(
            groups,
            graph,
        )

        # --------------------------------------------------
        # Step 5
        # --------------------------------------------------

        paragraph = self._build_paragraph(
            ordered
        )

        print()
        print(
            f"Final Sentences : "
            f"{len(ordered)}"
        )

        print(
            f"Paragraph Length : "
            f"{len(paragraph.split())} words"
        )

        print(
            f"Embedding Reuses : "
            f"{self.embedding_reuses}"
        )

        print(
            f"Embedding Fallbacks : "
            f"{self.embedding_fallbacks}"
        )

        print("=" * 70)

        return {
            "paragraph": paragraph,
            "graph": graph,
            "groups": groups,
            "entities": dict(
                self.entities
            ),
            "sentence_count": len(
                ordered
            ),
            "embedding_reuses": (
                self.embedding_reuses
            ),
            "embedding_fallbacks": (
                self.embedding_fallbacks
            ),
        }

    # ======================================================
    # SENTENCE GRAPH
    # ======================================================

    def _build_sentence_graph(
        self,
        selected_sentences: List[Dict],
    ) -> List[Dict]:

        nodes: List[Dict] = []

        # --------------------------------------------------
        # Create nodes
        # --------------------------------------------------

        for idx, item in enumerate(
            selected_sentences
        ):

            sentence = clean_sentence(
                item.get("text", "")
            )

            semantic = float(
                item.get(
                    "semantic",
                    item.get(
                        "similarity",
                        0.0,
                    ),
                )
            )

            relevance = float(
                item.get(
                    "relevance",
                    item.get(
                        "target_score",
                        semantic,
                    ),
                )
            )

            targets = list(
                item.get(
                    "targets",
                    [],
                )
            )

            entities = self._extract_entities(
                sentence
            )

            for entity in entities:

                self.entities[
                    entity
                ].append(idx)

            nodes.append(
                {
                    "id": idx,
                    "text": sentence,
                    "similarity": semantic,
                    "relevance": relevance,
                    "targets": targets,
                    "entities": entities,
                    "neighbors": [],
                }
            )

        # --------------------------------------------------
        # Build pairwise similarity matrix
        #
        # IMPORTANT:
        # No model inference happens here when reusable
        # embeddings are available.
        # --------------------------------------------------

        embeddings = []

        valid_nodes = []

        for node in nodes:

            embedding = (
                self.sentence_embeddings.get(
                    node["text"]
                )
            )

            if embedding is None:
                self.embedding_fallbacks += 1

                embedding = (
                    self.embedder.encode(
                        node["text"],
                        convert_to_tensor=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                )

            else:
                self.embedding_reuses += 1

            embeddings.append(
                embedding
            )

            valid_nodes.append(node)

        if embeddings:

            try:

                matrix = util.cos_sim(
                    embeddings,
                    embeddings,
                )

                matrix_np = (
                    matrix.detach()
                    .cpu()
                    .numpy()
                )

            except Exception:

                matrix_np = np.eye(
                    len(embeddings)
                )

        else:

            matrix_np = np.empty(
                (0, 0)
            )

        # --------------------------------------------------
        # Edges
        # --------------------------------------------------

        semantic_threshold = 0.55

        for i in range(
            len(nodes)
        ):

            for j in range(
                i + 1,
                len(nodes),
            ):

                entity_i = set(
                    nodes[i]["entities"]
                )

                entity_j = set(
                    nodes[j]["entities"]
                )

                entity_overlap = 0.0

                if entity_i or entity_j:

                    union = (
                        entity_i
                        | entity_j
                    )

                    intersection = (
                        entity_i
                        & entity_j
                    )

                    if union:
                        entity_overlap = (
                            len(intersection)
                            / len(union)
                        )

                semantic = float(
                    matrix_np[i][j]
                )

                target_overlap = len(
                    set(
                        nodes[i]["targets"]
                    )
                    & set(
                        nodes[j]["targets"]
                    )
                )

                # Evidence relation:
                #
                # semantic similarity
                # +
                # entity overlap
                # +
                # shared answer target
                #
                weight = (
                    0.60 * semantic
                    + 0.20 * entity_overlap
                    + 0.20 * min(
                        target_overlap,
                        1,
                    )
                )

                if (
                    weight >= semantic_threshold
                    or entity_overlap > 0.0
                ):

                    nodes[i][
                        "neighbors"
                    ].append(
                        {
                            "id": j,
                            "weight": float(
                                weight
                            ),
                        }
                    )

                    nodes[j][
                        "neighbors"
                    ].append(
                        {
                            "id": i,
                            "weight": float(
                                weight
                            ),
                        }
                    )

        edge_count = sum(
            len(node["neighbors"])
            for node in nodes
        ) // 2

        print()
        print("Sentence Graph")
        print(
            f"Nodes : {len(nodes)}"
        )
        print(
            f"Entities : {len(self.entities)}"
        )
        print(
            f"Edges : {edge_count}"
        )

        return nodes

    # ======================================================
    # ENTITY EXTRACTION
    # ======================================================

    def _extract_entities(
        self,
        sentence: str,
    ) -> List[str]:

        if not sentence:
            return []

        entities = []

        # --------------------------------------------------
        # spaCy when available
        # --------------------------------------------------

        if nlp is not None:

            try:

                doc = nlp(
                    sentence
                )

                for ent in doc.ents:

                    value = normalize_entity(
                        ent.text
                    )

                    if value:
                        entities.append(
                            value
                        )

            except Exception:
                pass

        # --------------------------------------------------
        # Lightweight fallback / domain entities
        # --------------------------------------------------

        known_patterns = (
            r"\bchandrayaan-3\b",
            r"\bchandrayaan-1\b",
            r"\bisro\b",
            r"\bnasa\b",
            r"\bcNSA\b",
            r"\bCNSA\b",
            r"\bvikram\b",
            r"\bpragyan\b",
            r"\bmoon\b",
            r"\blunar south pole\b",
        )

        for pattern in known_patterns:

            matches = re.findall(
                pattern,
                sentence,
                flags=re.IGNORECASE,
            )

            for match in matches:

                value = normalize_entity(
                    match
                )

                if value:
                    entities.append(
                        value
                    )

        # Unique while preserving order.
        unique = []

        seen = set()

        for entity in entities:

            if entity in seen:
                continue

            seen.add(entity)
            unique.append(entity)

        return unique

    # ======================================================
    # ENTITY GROUPING
    # ======================================================

    def _merge_entities(
        self,
        nodes: List[Dict],
    ) -> List[Dict]:

        print()
        print("Entity Merge")

        if not nodes:
            return []

        merged = []

        visited = set()

        for node in nodes:

            if node["id"] in visited:
                continue

            group_nodes = [
                node
            ]

            visited.add(
                node["id"]
            )

            # Direct neighbours only.
            for neighbor in node[
                "neighbors"
            ]:

                neighbor_id = neighbor[
                    "id"
                ]

                if neighbor_id in visited:
                    continue

                if neighbor[
                    "weight"
                ] < 0.55:
                    continue

                group_nodes.append(
                    nodes[neighbor_id]
                )

                visited.add(
                    neighbor_id
                )

            entity_set = set()

            target_set = set()

            for item in group_nodes:

                entity_set.update(
                    item["entities"]
                )

                target_set.update(
                    item["targets"]
                )

            merged.append(
                {
                    "group_id": len(
                        merged
                    ),
                    "sentences": [
                        item["text"]
                        for item in group_nodes
                    ],
                    "entities": sorted(
                        entity_set
                    ),
                    "targets": sorted(
                        target_set
                    ),
                    "similarity": max(
                        item["similarity"]
                        for item in group_nodes
                    ),
                    "relevance": max(
                        item["relevance"]
                        for item in group_nodes
                    ),
                }
            )

        print(
            f"Groups Created : "
            f"{len(merged)}"
        )

        return merged

    # ======================================================
    # REDUNDANCY REMOVAL
    # ======================================================

    def _remove_redundancy(
        self,
        groups: List[Dict],
        similarity_threshold: float = 0.90,
    ) -> List[Dict]:

        print()
        print(
            "Semantic Redundancy Removal"
        )

        cleaned_groups = []

        removed = 0

        for group in groups:

            sentences = group[
                "sentences"
            ]

            if len(sentences) <= 1:

                cleaned_groups.append(
                    group
                )

                continue

            # --------------------------------------------------
            # Reuse embeddings.
            # --------------------------------------------------

            embeddings = []

            for sentence in sentences:

                embedding = (
                    self.sentence_embeddings.get(
                        sentence
                    )
                )

                if embedding is None:

                    self.embedding_fallbacks += 1

                    embedding = (
                        self.embedder.encode(
                            sentence,
                            convert_to_tensor=True,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                        )
                    )

                else:

                    self.embedding_reuses += 1

                embeddings.append(
                    embedding
                )

            # --------------------------------------------------
            # Greedy redundancy filtering.
            # --------------------------------------------------

            keep = []

            keep_embeddings = []

            for sentence, embedding in zip(
                sentences,
                embeddings,
            ):

                duplicate = False

                for existing in keep_embeddings:

                    similarity = float(
                        util.cos_sim(
                            embedding,
                            existing,
                        ).item()
                    )

                    if (
                        similarity
                        >= similarity_threshold
                    ):

                        duplicate = True
                        removed += 1
                        break

                if not duplicate:

                    keep.append(
                        sentence
                    )

                    keep_embeddings.append(
                        embedding
                    )

            group["sentences"] = keep

            if keep:
                cleaned_groups.append(
                    group
                )

        print(
            f"Removed : {removed}"
        )

        print(
            f"Remaining Groups : "
            f"{len(cleaned_groups)}"
        )

        return cleaned_groups

    # ======================================================
    # ORDERING
    # ======================================================

    def _order_sentences(
        self,
        groups: List[Dict],
        graph: List[Dict],
    ) -> List[str]:

        print()
        print(
            "Question-aware Evidence Ordering"
        )

        graph_lookup = {
            node["text"]: node
            for node in graph
        }

        scored_groups = []

        for group in groups:

            connectivity = 0.0
            target_count = len(
                group.get(
                    "targets",
                    [],
                )
            )

            for sentence in group[
                "sentences"
            ]:

                node = graph_lookup.get(
                    sentence
                )

                if node:

                    connectivity += (
                        sum(
                            neighbor[
                                "weight"
                            ]
                            for neighbor
                            in node[
                                "neighbors"
                            ]
                        )
                    )

            centrality = (
                connectivity
                / max(
                    len(
                        group[
                            "sentences"
                        ]
                    ),
                    1,
                )
            )

            # Normalize connectivity.
            centrality = min(
                centrality / 2.0,
                1.0,
            )

            target_score = min(
                target_count / 2.0,
                1.0,
            )

            score = (
                0.40 * group[
                    "relevance"
                ]
                + 0.35 * target_score
                + 0.15 * group[
                    "similarity"
                ]
                + 0.10 * centrality
            )

            scored_groups.append(
                (
                    score,
                    group,
                )
            )

        scored_groups.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        ordered = []

        visited = set()

        for _, group in scored_groups:

            for sentence in group[
                "sentences"
            ]:

                if sentence in visited:
                    continue

                ordered.append(
                    sentence
                )

                visited.add(
                    sentence
                )

        print(
            f"Ordered "
            f"{len(ordered)} "
            f"evidence sentences"
        )

        return ordered

    # ======================================================
    # PARAGRAPH BUILDER
    # ======================================================

    def _build_paragraph(
        self,
        ordered_sentences: List[str],
    ) -> str:

        print()
        print(
            "Paragraph Builder"
        )

        cleaned = []

        for sentence in ordered_sentences:

            sentence = clean_sentence(
                sentence
            )

            if not sentence:
                continue

            cleaned.append(
                sentence
            )

        paragraph = " ".join(
            cleaned
        ).strip()

        if (
            paragraph
            and not paragraph.endswith(".")
        ):
            paragraph += "."

        print(
            f"Paragraph Length : "
            f"{len(paragraph.split())} words"
        )

        return paragraph

    # ======================================================
    # DEBUG
    # ======================================================

    def print_graph(
        self,
        graph: List[Dict],
    ):

        print()
        print("=" * 60)
        print("Evidence Graph")
        print("=" * 60)

        for node in graph:

            print()
            print(
                f"Node {node['id']}"
            )

            print(
                f"Similarity : "
                f"{node['similarity']:.4f}"
            )

            print(
                f"Relevance : "
                f"{node['relevance']:.4f}"
            )

            print(
                f"Targets : "
                f"{node['targets']}"
            )

            print(
                f"Entities : "
                f"{node['entities']}"
            )

            print(
                f"Neighbors : "
                f"{node['neighbors']}"
            )