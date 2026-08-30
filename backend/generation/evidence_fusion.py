"""
Phase 10.2
Evidence Fusion

Quality-aware extractive fusion.

Optimization #18:
- reuse Phase 10.1 sentence embeddings
- build one embedding matrix
- compute one similarity matrix
- never invent facts

Quality improvements:
- preserve Phase 10.1 answer-bearing ordering
- never replace selected evidence with generic graph-central
  sentences
- graph ordering is only allowed to reorder already selected
  evidence
- malformed text is not introduced during fusion
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch

from backend.models.model_registry import ModelRegistry


EDGE_THRESHOLD = 0.45
REDUNDANCY_THRESHOLD = 0.92


def _clamp_similarity(value: float) -> float:
    return max(
        0.0,
        min(1.0, float(value)),
    )


def _as_tensor(embeddings: Any) -> Optional[torch.Tensor]:
    if embeddings is None:
        return None

    if isinstance(embeddings, torch.Tensor):
        return embeddings

    try:
        return torch.as_tensor(embeddings)
    except Exception:
        return None


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


class EvidenceFusion:

    def __init__(
        self,
        sentence_embeddings: Optional[Dict[str, Any]] = None,
    ):
        self.graph = defaultdict(list)
        self.entities = defaultdict(list)

        self.embedder = (
            ModelRegistry.get_embedding_model()
        )

        self.sentence_embeddings = sentence_embeddings

        self.embedding_matrix = None
        self.embedding_lookup: Dict[str, int] = {}

        self.embedding_passes = 0
        self.embedding_fallbacks = 0
        self.embedding_reuses = 0

        self.pairwise_comparisons = 0
        self.matrix_similarity_computed = False

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

        print()
        print("=" * 70)
        print("Phase 10.2 : Evidence Fusion")
        print("=" * 70)
        print(
            f"Input Sentences : {len(selected)}"
        )

        self.graph = defaultdict(list)
        self.entities = defaultdict(list)
        self.embedding_matrix = None
        self.embedding_lookup = {}
        self.embedding_passes = 0
        self.embedding_fallbacks = 0
        self.embedding_reuses = 0
        self.pairwise_comparisons = 0
        self.matrix_similarity_computed = False

        self._prepare_embedding_matrix(
            generation_result,
            selected,
            self.sentence_embeddings,
        )

        graph = self._build_sentence_graph(
            selected
        )

        groups = self._merge_entities(
            graph
        )

        groups = self._remove_redundancy(
            groups
        )

        # Important quality rule:
        # Phase 10.1 has already selected answer-bearing evidence.
        # Do not allow graph centrality to introduce or prioritize
        # unrelated sentences.
        ordered = self._order_sentences(
            groups,
            graph,
            selected,
        )

        paragraph = self._build_paragraph(
            ordered
        )

        print(
            f"Graph Nodes : {len(graph)}"
        )
        print(
            f"Embedding Reuses : {self.embedding_reuses}"
        )
        print(
            f"Embedding Passes : {self.embedding_passes}"
        )
        print(
            f"Embedding Fallbacks : {self.embedding_fallbacks}"
        )
        print(
            f"Pairwise Comparisons : {self.pairwise_comparisons}"
        )
        print(
            "Similarity Matrix : "
            + (
                "USED"
                if self.matrix_similarity_computed
                else "NOT USED"
            )
        )
        print("=" * 70)

        return {
            "paragraph": paragraph,
            "graph": graph,
            "groups": groups,
            "entities": dict(self.entities),
            "sentence_count": len(ordered),
            "embedding_reuse_enabled": True,
            "embedding_passes": self.embedding_passes,
            "embedding_fallbacks": self.embedding_fallbacks,
            "embedding_reuses": self.embedding_reuses,
            "pairwise_comparisons": self.pairwise_comparisons,
            "matrix_similarity_computed": (
                self.matrix_similarity_computed
            ),
        }

    # ======================================================
    # EMBEDDING MATRIX
    # ======================================================

    def _prepare_embedding_matrix(
        self,
        generation_result: Dict,
        selected_sentences: List[Dict],
        sentence_embeddings: Optional[Dict[str, Any]] = None,
    ) -> None:

        if not selected_sentences:
            return

        selected_texts = [
            _safe_text(item.get("text", ""))
            for item in selected_sentences
        ]

        selected_texts = [
            text
            for text in selected_texts
            if text
        ]

        if not selected_texts:
            return

        reusable = (
            sentence_embeddings
            if sentence_embeddings is not None
            else generation_result.get(
                "sentence_embeddings"
            )
        )

        if isinstance(reusable, dict) and reusable:

            ordered_embeddings = []

            for text in selected_texts:

                tensor = _as_tensor(
                    reusable.get(text)
                )

                if tensor is None:
                    ordered_embeddings = []
                    break

                ordered_embeddings.append(
                    tensor
                )

            if (
                ordered_embeddings
                and len(ordered_embeddings)
                == len(selected_texts)
            ):

                self.embedding_matrix = torch.stack(
                    ordered_embeddings
                )

                if self.embedding_matrix.dim() == 1:
                    self.embedding_matrix = (
                        self.embedding_matrix.unsqueeze(0)
                    )

                self.embedding_matrix = (
                    torch.nn.functional.normalize(
                        self.embedding_matrix,
                        p=2,
                        dim=1,
                    )
                )

                self.embedding_lookup = {
                    text: index
                    for index, text in enumerate(
                        selected_texts
                    )
                }

                self.embedding_reuses = len(
                    selected_texts
                )

                print(
                    "[Optimization #18] "
                    "Reusing #17 sentence embeddings"
                )
                print(
                    "[Optimization #18] "
                    f"Reusable embeddings : "
                    f"{self.embedding_reuses}"
                )
                return

        # Compatibility fallback.
        embeddings = self.embedder.encode(
            selected_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        self.embedding_passes = 1
        self.embedding_fallbacks = 1

        self.embedding_matrix = _as_tensor(
            embeddings
        )

        if self.embedding_matrix is None:
            return

        if self.embedding_matrix.dim() == 1:
            self.embedding_matrix = (
                self.embedding_matrix.unsqueeze(0)
            )

        self.embedding_matrix = (
            torch.nn.functional.normalize(
                self.embedding_matrix,
                p=2,
                dim=1,
            )
        )

        self.embedding_lookup = {
            text: index
            for index, text in enumerate(
                selected_texts
            )
        }

    # ======================================================
    # SIMILARITY MATRIX
    # ======================================================

    def _compute_similarity_matrix(self):
        if self.embedding_matrix is None:
            return None

        if self.embedding_matrix.numel() == 0:
            return None

        matrix = torch.matmul(
            self.embedding_matrix,
            self.embedding_matrix.transpose(0, 1),
        )

        matrix = torch.clamp(
            matrix,
            min=-1.0,
            max=1.0,
        )

        self.matrix_similarity_computed = True

        n = int(
            self.embedding_matrix.shape[0]
        )

        self.pairwise_comparisons = (
            n * (n - 1) // 2
        )

        return matrix

    # ======================================================
    # SENTENCE GRAPH
    # ======================================================

    def _build_sentence_graph(
        self,
        selected_sentences: List[Dict],
    ) -> List[Dict]:

        nodes: List[Dict] = []

        if not selected_sentences:
            return nodes

        nlp = ModelRegistry.get_nlp()

        for idx, item in enumerate(
            selected_sentences
        ):

            sentence = _safe_text(
                item.get("text", "")
            )

            doc = nlp(sentence)

            entity_names = []

            for ent in doc.ents:

                name = ent.text.strip()

                if not name:
                    continue

                if name not in entity_names:
                    entity_names.append(name)

                self.entities[name].append(idx)

            nodes.append(
                {
                    "id": idx,
                    "text": sentence,
                    "similarity": float(
                        item.get(
                            "similarity",
                            item.get(
                                "semantic",
                                0.0,
                            ),
                        )
                    ),
                    "target_score": float(
                        item.get(
                            "target_score",
                            0.0,
                        )
                    ),
                    "quality": float(
                        item.get(
                            "quality",
                            1.0,
                        )
                    ),
                    "targets": list(
                        item.get(
                            "targets",
                            [],
                        )
                    ),
                    "entities": entity_names,
                    "neighbors": [],
                }
            )

        similarity_matrix = (
            self._compute_similarity_matrix()
        )

        for i in range(len(nodes)):

            entities_i = set(
                nodes[i]["entities"]
            )

            for j in range(
                i + 1,
                len(nodes),
            ):

                entities_j = set(
                    nodes[j]["entities"]
                )

                overlap = self._entity_overlap(
                    list(entities_i),
                    list(entities_j),
                )

                semantic = 0.0

                if similarity_matrix is not None:
                    semantic = _clamp_similarity(
                        similarity_matrix[
                            i, j
                        ].item()
                    )

                weight = (
                    0.6 * overlap
                    + 0.4 * semantic
                )

                if weight > EDGE_THRESHOLD:

                    nodes[i]["neighbors"].append(
                        {
                            "id": j,
                            "weight": float(weight),
                        }
                    )

                    nodes[j]["neighbors"].append(
                        {
                            "id": i,
                            "weight": float(weight),
                        }
                    )

        print()
        print("Sentence Graph")
        print(
            f"Nodes : {len(nodes)}"
        )
        print(
            f"Entities : {len(self.entities)}"
        )

        edge_count = (
            sum(
                len(node["neighbors"])
                for node in nodes
            )
            // 2
        )

        print(
            f"Edges : {edge_count}"
        )

        return nodes

    # ======================================================
    # ENTITY GROUPS
    # ======================================================

    def _merge_entities(
        self,
        nodes: List[Dict],
    ) -> List[Dict]:

        print()
        print("Entity Merge")

        merged = []
        visited = set()

        for node in nodes:

            node_id = node["id"]

            if node_id in visited:
                continue

            group = [node]
            visited.add(node_id)

            for neighbor in node["neighbors"]:

                neighbor_id = neighbor["id"]

                if neighbor_id in visited:
                    continue

                if neighbor["weight"] < EDGE_THRESHOLD:
                    continue

                group.append(
                    nodes[neighbor_id]
                )

                visited.add(neighbor_id)

            entity_set = set()

            for item in group:
                entity_set.update(
                    item["entities"]
                )

            merged.append(
                {
                    "group_id": len(merged),
                    "sentences": [
                        item["text"]
                        for item in group
                    ],
                    "entities": sorted(
                        entity_set
                    ),
                    "similarity": max(
                        (
                            item["similarity"]
                            for item in group
                        ),
                        default=0.0,
                    ),
                }
            )

        print(
            f"Groups Created : {len(merged)}"
        )

        return merged

    # ======================================================
    # REDUNDANCY REMOVAL
    # ======================================================

    def _remove_redundancy(
        self,
        groups: List[Dict],
        similarity_threshold: float = REDUNDANCY_THRESHOLD,
    ) -> List[Dict]:

        print()
        print("Semantic Redundancy Removal")

        cleaned_groups = []
        removed = 0

        for group in groups:

            sentences = group["sentences"]

            if len(sentences) <= 1:
                cleaned_groups.append(group)
                continue

            keep = []
            keep_indices = []

            for sentence in sentences:

                current_index = (
                    self.embedding_lookup.get(
                        sentence
                    )
                )

                duplicate = False

                if (
                    current_index is not None
                    and self.embedding_matrix is not None
                ):

                    for existing_index in keep_indices:

                        similarity = torch.dot(
                            self.embedding_matrix[
                                current_index
                            ],
                            self.embedding_matrix[
                                existing_index
                            ],
                        ).item()

                        similarity = _clamp_similarity(
                            similarity
                        )

                        if similarity >= similarity_threshold:
                            duplicate = True
                            removed += 1
                            break

                if not duplicate:
                    keep.append(sentence)

                    if current_index is not None:
                        keep_indices.append(
                            current_index
                        )

            if not keep and sentences:
                keep = [sentences[0]]

            group["sentences"] = keep
            cleaned_groups.append(group)

        print(
            f"Removed : {removed}"
        )
        print(
            f"Remaining Groups : "
            f"{len(cleaned_groups)}"
        )

        return cleaned_groups

    # ======================================================
    # ORDER SENTENCES
    # ======================================================

    def _order_sentences(
        self,
        groups: List[Dict],
        graph: List[Dict],
        selected_sentences: List[Dict],
    ) -> List[str]:

        print()
        print("Graph-aware Ordering")

        # --------------------------------------------------
        # Preserve Phase 10.1 answer-bearing order.
        #
        # Graph ordering is secondary. This prevents a highly
        # connected but generic sentence from replacing a direct
        # definition sentence.
        # --------------------------------------------------

        selected_order = [
            _safe_text(item.get("text", ""))
            for item in selected_sentences
            if _safe_text(item.get("text", ""))
        ]

        graph_lookup = {
            node["text"]: node
            for node in graph
        }

        ordered: List[str] = []

        # First pass: selected evidence in Phase 10.1 order.
        for sentence in selected_order:
            if sentence not in ordered:
                ordered.append(sentence)

        # Second pass: any surviving group sentence not already
        # present. This is only a safety fallback.
        for group in groups:
            for sentence in group["sentences"]:
                if sentence not in ordered:
                    ordered.append(sentence)

        print(
            f"Ordered {len(ordered)} evidence sentences"
        )

        return ordered

    # ======================================================
    # PARAGRAPH
    # ======================================================

    def _build_paragraph(
        self,
        ordered_sentences: List[str],
    ) -> str:

        print()
        print("Paragraph Builder")

        paragraph = " ".join(
            sentence.strip()
            for sentence in ordered_sentences
            if sentence.strip()
        )

        paragraph = " ".join(
            paragraph.split()
        )

        if paragraph and not paragraph.endswith("."):
            paragraph += "."

        print(
            f"Paragraph Length : "
            f"{len(paragraph.split())} words"
        )

        return paragraph

    # ======================================================
    # HELPERS
    # ======================================================

    def _entity_overlap(
        self,
        entities_a: List[str],
        entities_b: List[str],
    ) -> float:

        set_a = set(entities_a)
        set_b = set(entities_b)

        if not set_a and not set_b:
            return 0.0

        union = set_a | set_b

        if not union:
            return 0.0

        return len(set_a & set_b) / len(union)

    def _semantic_similarity(
        self,
        sentence_a: str,
        sentence_b: str,
    ) -> float:

        if not sentence_a or not sentence_b:
            return 0.0

        index_a = self.embedding_lookup.get(
            sentence_a
        )
        index_b = self.embedding_lookup.get(
            sentence_b
        )

        if (
            index_a is None
            or index_b is None
            or self.embedding_matrix is None
        ):
            return 0.0

        similarity = torch.dot(
            self.embedding_matrix[index_a],
            self.embedding_matrix[index_b],
        ).item()

        return _clamp_similarity(
            similarity
        )

    def _edge_weight(
        self,
        entities_a: List[str],
        entities_b: List[str],
        sentence_a: str,
        sentence_b: str,
    ) -> float:

        overlap = self._entity_overlap(
            entities_a,
            entities_b,
        )

        semantic = self._semantic_similarity(
            sentence_a,
            sentence_b,
        )

        return (
            0.6 * overlap
            + 0.4 * semantic
        )

    def _graph_centrality(
        self,
        neighbors: List[Dict],
    ) -> float:

        return sum(
            float(
                neighbor.get(
                    "weight",
                    0.0,
                )
            )
            for neighbor in neighbors
        )
