"""
Phase 10.2
Evidence Fusion

Optimization #18
----------------
Vectorized Evidence Fusion / Reusable Embedding Matrix

This module receives the question-aware evidence selected by
Phase 10.1 and converts it into a structured evidence graph
before building the final extractive paragraph.

Optimization #18 removes repeated sentence-pair embedding
inference.

OLD:
    sentence_a + sentence_b
        ↓
    BGE.encode(...)
        ↓
    cosine similarity

    repeated for every pair

NEW:
    selected sentence embeddings
        ↓
    one reusable embedding matrix
        ↓
    cosine similarity matrix
        ↓
    cheap tensor lookups

Optimization #17 compatibility
-------------------------------
Phase 10.1 already returns:

    generation_result["sentence_embeddings"]

This module reuses those embeddings directly.

If reusable embeddings are unavailable, a single batch encoding
is performed as a compatibility fallback.

Design goals
------------
1. Never invent facts.
2. Preserve the extractive nature of the pipeline.
3. Reuse Optimization #17 sentence embeddings.
4. Avoid repeated BGE inference.
5. Compute pairwise semantic similarity with matrix operations.
6. Preserve existing EvidenceFusion output compatibility.
7. Preserve entity extraction and graph construction.
8. Preserve redundancy removal behavior.
9. Keep diagnostics useful for profiling.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from sentence_transformers import util

from backend.models.model_registry import ModelRegistry


# ==========================================================
# CONSTANTS
# ==========================================================

EDGE_THRESHOLD = 0.45
REDUNDANCY_THRESHOLD = 0.92

# Optimization #18:
# Similarity values are computed from normalized embeddings.
SIMILARITY_MIN = 0.0
SIMILARITY_MAX = 1.0


# ==========================================================
# spaCy MODEL
# ==========================================================

nlp = ModelRegistry.get_nlp()


# ==========================================================
# HELPERS
# ==========================================================

def _clamp_similarity(
    value: float,
) -> float:
    """
    Clamp cosine similarity into [0, 1].

    BGE embeddings are normalized, so cosine similarity is
    normally in [-1, 1]. EvidenceFusion historically treated
    similarity as a non-negative evidence score, therefore
    negative similarities are clipped to zero.
    """

    return max(
        SIMILARITY_MIN,
        min(
            SIMILARITY_MAX,
            float(value),
        ),
    )


def _as_tensor(
    embeddings: Any,
) -> Optional[torch.Tensor]:
    """
    Convert an embedding container into a torch Tensor.

    Supports:
        torch.Tensor
        numpy arrays
        lists

    Returns
    -------
    torch.Tensor or None
    """

    if embeddings is None:
        return None

    if isinstance(
        embeddings,
        torch.Tensor,
    ):
        return embeddings

    try:
        return torch.as_tensor(
            embeddings
        )
    except Exception:
        return None


# ==========================================================
# EVIDENCE FUSION
# ==========================================================

class EvidenceFusion:
    """
    Phase 10.2 Evidence Fusion Engine.

    Optimization #18:
        Vectorized semantic similarity using a reusable
        sentence embedding matrix.

    Input
    -----
    generate_answer() result:

        {
            "answer": "...",
            "selected_sentences": [
                {
                    "rank": 1,
                    "text": "...",
                    "similarity": 0.91,
                    ...
                }
            ],
            "sentence_embeddings": {
                "sentence text": embedding
            }
        }

    Output
    ------

        {
            "paragraph": "...",
            "graph": ...,
            "groups": ...,
            "entities": ...,
            "sentence_count": ...,
            "embedding_reuse_enabled": True,
            "embedding_passes": 0 or 1,
            "embedding_fallbacks": 0 or 1
        }
    """

    def __init__(
        self,
        sentence_embeddings: Optional[Dict[str, Any]] = None,
    ):
        self.graph = defaultdict(list)
        self.entities = defaultdict(list)

        # --------------------------------------------------
        # Optimization #1:
        # Centralized model registry.
        # --------------------------------------------------

        self.embedder = (
            ModelRegistry.get_embedding_model()
        )

        # --------------------------------------------------
        # Optimization #18:
        # Receive reusable sentence embeddings from
        # Optimization #17.
        #
        # run_rerank.py already passes:
        #
        #     sentence_embeddings=sentence_embeddings
        #
        # --------------------------------------------------

        self.sentence_embeddings = (
            sentence_embeddings
        )

        # --------------------------------------------------
        # Optimization #18 diagnostics.
        # --------------------------------------------------

        self.embedding_matrix: Optional[
            torch.Tensor
        ] = None

        self.embedding_lookup: Dict[
            str,
            int,
        ] = {}

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
        """
        Main entry point.

        Executes:

            reusable embeddings
                    ↓
            sentence graph
                    ↓
            entity grouping
                    ↓
            redundancy removal
                    ↓
            evidence ordering
                    ↓
            paragraph construction
        """

        selected = (
            generation_result.get(
                "selected_sentences",
                [],
            )
        )

        print()
        print("=" * 70)
        print("Phase 10.2 : Evidence Fusion")
        print("=" * 70)

        print(
            f"Input Sentences : "
            f"{len(selected)}"
        )

        # --------------------------------------------------
        # Reset per-query state.
        # --------------------------------------------------

        self.graph = defaultdict(list)
        self.entities = defaultdict(list)

        self.embedding_matrix = None
        self.embedding_lookup = {}

        self.embedding_passes = 0
        self.embedding_fallbacks = 0
        self.embedding_reuses = 0

        self.pairwise_comparisons = 0
        self.matrix_similarity_computed = False

        # --------------------------------------------------
        # Optimization #18
        #
        # Prepare one embedding matrix BEFORE graph
        # construction.
        #
        # This matrix is reused by:
        #
        #   graph construction
        #   redundancy removal
        #
        # --------------------------------------------------

        self._prepare_embedding_matrix(
            generation_result,
            selected,
            sentence_embeddings=self.sentence_embeddings,
        )

        # --------------------------------------------------
        # Step 1
        # Sentence Graph
        # --------------------------------------------------

        graph = self._build_sentence_graph(
            selected
        )

        # --------------------------------------------------
        # Step 2
        # Entity Groups
        # --------------------------------------------------

        groups = self._merge_entities(
            graph
        )

        # --------------------------------------------------
        # Step 3
        # Remove Redundancy
        # --------------------------------------------------

        groups = self._remove_redundancy(
            groups
        )

        # --------------------------------------------------
        # Step 4
        # Order Sentences
        # --------------------------------------------------

        ordered = self._order_sentences(
            groups,
            graph,
        )

        # --------------------------------------------------
        # Step 5
        # Build Paragraph
        # --------------------------------------------------

        paragraph = self._build_paragraph(
            ordered
        )

        print()
        print(
            f"Graph Nodes : {len(graph)}"
        )

        print(
            f"Embedding Reuses : "
            f"{self.embedding_reuses}"
        )

        print(
            f"Embedding Passes : "
            f"{self.embedding_passes}"
        )

        print(
            f"Embedding Fallbacks : "
            f"{self.embedding_fallbacks}"
        )

        print(
            f"Pairwise Comparisons : "
            f"{self.pairwise_comparisons}"
        )

        print(
            f"Similarity Matrix : "
            f"{'USED' if self.matrix_similarity_computed else 'NOT USED'}"
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

            # --------------------------------------------------
            # Optimization #18 diagnostics
            # --------------------------------------------------

            "embedding_reuse_enabled": True,

            "embedding_passes": (
                self.embedding_passes
            ),

            "embedding_fallbacks": (
                self.embedding_fallbacks
            ),

            "embedding_reuses": (
                self.embedding_reuses
            ),

            "pairwise_comparisons": (
                self.pairwise_comparisons
            ),

            "matrix_similarity_computed": (
                self.matrix_similarity_computed
            ),
        }

    # ======================================================
    # OPTIMIZATION #18
    # EMBEDDING MATRIX PREPARATION
    # ======================================================

    def _prepare_embedding_matrix(
        self,
        generation_result: Dict,
        selected_sentences: List[Dict],
        sentence_embeddings: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Prepare one reusable embedding matrix.

        Preferred source
        ----------------
        Optimization #17 output:

            generation_result["sentence_embeddings"]

        This is a dictionary:

            {
                sentence_text: embedding
            }

        We extract only embeddings required by the selected
        sentences.

        Fallback
        --------
        If Optimization #17 embeddings are unavailable,
        encode all selected sentences ONCE in one batch.

        This fallback preserves compatibility with older
        callers while ensuring that even fallback mode does
        not perform pairwise inference.
        """

        if not selected_sentences:
            return

        selected_texts = [
            item.get(
                "text",
                "",
            )
            for item in selected_sentences
        ]

        selected_texts = [
            text
            for text in selected_texts
            if text
        ]

        if not selected_texts:
            return

        reusable_embeddings = (
            sentence_embeddings
            if sentence_embeddings is not None
            else generation_result.get(
                "sentence_embeddings"
            )
        )

        # --------------------------------------------------
        # Preferred path:
        # Reuse Optimization #17 embeddings.
        # --------------------------------------------------

        if reusable_embeddings:
            ordered_embeddings = []

            reusable_count = 0

            for text in selected_texts:

                embedding = (
                    reusable_embeddings.get(
                        text
                    )
                    if isinstance(
                        reusable_embeddings,
                        dict,
                    )
                    else None
                )

                tensor = _as_tensor(
                    embedding
                )

                if tensor is None:
                    ordered_embeddings = []
                    break

                ordered_embeddings.append(
                    tensor
                )

                reusable_count += 1

            if (
                ordered_embeddings
                and len(
                    ordered_embeddings
                )
                == len(selected_texts)
            ):

                self.embedding_matrix = (
                    torch.stack(
                        ordered_embeddings
                    )
                )

                # Ensure 2-D matrix.
                if (
                    self.embedding_matrix.dim()
                    == 1
                ):
                    self.embedding_matrix = (
                        self.embedding_matrix.unsqueeze(
                            0
                        )
                    )

                # Normalize once.
                self.embedding_matrix = (
                    torch.nn.functional.normalize(
                        self.embedding_matrix,
                        p=2,
                        dim=1,
                    )
                )

                self.embedding_lookup = {
                    text: index
                    for index, text
                    in enumerate(
                        selected_texts
                    )
                }

                self.embedding_reuses = (
                    reusable_count
                )

                print(
                    "[Optimization #18] "
                    "Reusing #17 sentence embeddings"
                )

                print(
                    f"[Optimization #18] "
                    f"Reusable embeddings : "
                    f"{reusable_count}"
                )

                return

        # --------------------------------------------------
        # Compatibility fallback:
        #
        # One batch encode.
        #
        # IMPORTANT:
        # This is still only ONE model inference pass.
        # --------------------------------------------------

        print(
            "[Optimization #18] "
            "Reusable embeddings unavailable"
        )

        print(
            "[Optimization #18] "
            "Executing one fallback batch embedding pass"
        )

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

        if (
            self.embedding_matrix.dim()
            == 1
        ):
            self.embedding_matrix = (
                self.embedding_matrix.unsqueeze(
                    0
                )
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
            for index, text
            in enumerate(
                selected_texts
            )
        }

    # ======================================================
    # OPTIMIZATION #18
    # SIMILARITY MATRIX
    # ======================================================

    def _compute_similarity_matrix(
        self,
    ) -> Optional[torch.Tensor]:
        """
        Compute the full sentence similarity matrix.

        Because embeddings are normalized:

            E @ E.T

        is equivalent to cosine similarity.

        Example:

            E =
                [E1]
                [E2]
                [E3]

        produces:

                E1·E1  E1·E2  E1·E3
                E2·E1  E2·E2  E2·E3
                E3·E1  E3·E2  E3·E3

        This replaces repeated calls to:

            embedder.encode(
                [sentence_a, sentence_b]
            )

        """

        if self.embedding_matrix is None:
            return None

        if (
            self.embedding_matrix.numel()
            == 0
        ):
            return None

        similarity_matrix = (
            torch.matmul(
                self.embedding_matrix,
                self.embedding_matrix.transpose(
                    0,
                    1,
                ),
            )
        )

        # Clamp numerical noise.
        similarity_matrix = (
            torch.clamp(
                similarity_matrix,
                min=-1.0,
                max=1.0,
            )
        )

        self.matrix_similarity_computed = True

        node_count = (
            self.embedding_matrix.shape[0]
        )

        self.pairwise_comparisons = (
            node_count
            * (node_count - 1)
            // 2
        )

        return similarity_matrix

    # ======================================================
    # STEP 1
    # SENTENCE GRAPH
    # ======================================================

    def _build_sentence_graph(
        self,
        selected_sentences: List[Dict],
    ) -> List[Dict]:
        """
        Builds the initial evidence graph.

        Every sentence becomes one node.

        Nodes contain:

            sentence
            similarity
            named entities
            outgoing links

        Edges use:

            entity overlap
            semantic similarity

        Optimization #18:
            semantic similarities come from ONE matrix.
        """

        nodes: List[Dict] = []

        if not selected_sentences:
            return nodes

        # --------------------------------------------------
        # Create nodes.
        # --------------------------------------------------

        for idx, item in enumerate(
            selected_sentences
        ):

            sentence = item.get(
                "text",
                "",
            )

            similarity = item.get(
                "similarity",
                0.0,
            )

            doc = nlp(sentence)

            entity_names: List[str] = []

            for ent in doc.ents:

                name = ent.text.strip()

                if not name:
                    continue

                if name not in entity_names:
                    entity_names.append(
                        name
                    )

                self.entities[
                    name
                ].append(idx)

            nodes.append(
                {
                    "id": idx,

                    "text": sentence,

                    "similarity": float(
                        similarity
                    ),

                    "entities": entity_names,

                    "neighbors": [],
                }
            )

        # --------------------------------------------------
        # Compute semantic similarity matrix ONCE.
        # --------------------------------------------------

        similarity_matrix = (
            self._compute_similarity_matrix()
        )

        # --------------------------------------------------
        # Connect nodes.
        # --------------------------------------------------

        for i in range(
            len(nodes)
        ):

            entities_i = set(
                nodes[i][
                    "entities"
                ]
            )

            for j in range(
                i + 1,
                len(nodes),
            ):

                entities_j = set(
                    nodes[j][
                        "entities"
                    ]
                )

                overlap = (
                    self._entity_overlap(
                        list(
                            entities_i
                        ),
                        list(
                            entities_j
                        ),
                    )
                )

                semantic = 0.0

                if (
                    similarity_matrix
                    is not None
                ):

                    semantic = (
                        float(
                            similarity_matrix[
                                i,
                                j,
                            ].item()
                        )
                    )

                    semantic = (
                        _clamp_similarity(
                            semantic
                        )
                    )

                weight = (
                    0.6 * overlap
                    + 0.4 * semantic
                )

                if (
                    weight
                    > EDGE_THRESHOLD
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

        # --------------------------------------------------
        # Debug
        # --------------------------------------------------

        print()
        print(
            "Sentence Graph"
        )

        print(
            f"Nodes : {len(nodes)}"
        )

        print(
            f"Entities : "
            f"{len(self.entities)}"
        )

        edge_count = (
            sum(
                len(
                    node[
                        "neighbors"
                    ]
                )
                for node in nodes
            )
            // 2
        )

        print(
            f"Edges : {edge_count}"
        )

        return nodes

    # ======================================================
    # STEP 2
    # ENTITY GROUPS
    # ======================================================

    def _merge_entities(
        self,
        nodes: List[Dict],
    ) -> List[Dict]:
        """
        Groups together sentences that discuss
        related entities.

        This does not modify sentence text.
        It creates logical evidence groups.
        """

        print()
        print(
            "Entity Merge"
        )

        merged: List[Dict] = []

        visited = set()

        for node in nodes:

            node_id = node[
                "id"
            ]

            if node_id in visited:
                continue

            group = [node]

            visited.add(
                node_id
            )

            # --------------------------------------------------
            # Find directly connected neighbours.
            # --------------------------------------------------

            for neighbor in node[
                "neighbors"
            ]:

                neighbor_id = neighbor[
                    "id"
                ]

                if (
                    neighbor[
                        "weight"
                    ]
                    < EDGE_THRESHOLD
                ):
                    continue

                if neighbor_id in visited:
                    continue

                group.append(
                    nodes[
                        neighbor_id
                    ]
                )

                visited.add(
                    neighbor_id
                )

            # --------------------------------------------------
            # Collect unique entities.
            # --------------------------------------------------

            entity_set = set()

            for item in group:

                entity_set.update(
                    item[
                        "entities"
                    ]
                )

            # --------------------------------------------------
            # Store group.
            # --------------------------------------------------

            merged.append(
                {
                    "group_id": len(
                        merged
                    ),

                    "sentences": [
                        g["text"]
                        for g in group
                    ],

                    "entities": sorted(
                        entity_set
                    ),

                    "similarity": max(
                        (
                            g[
                                "similarity"
                            ]
                            for g in group
                        ),
                        default=0.0,
                    ),
                }
            )

        print(
            f"Groups Created : "
            f"{len(merged)}"
        )

        return merged

    # ======================================================
    # STEP 3
    # REDUNDANCY REMOVAL
    # ======================================================

    def _remove_redundancy(
        self,
        groups: List[Dict],
        similarity_threshold: float = (
            REDUNDANCY_THRESHOLD
        ),
    ) -> List[Dict]:
        """
        Removes semantically redundant evidence.

        Optimization #18:
            Does NOT encode sentences.

        Instead it reuses the already prepared
        embedding matrix.

        Redundancy comparison becomes:

            similarity_matrix[i][j]

        instead of:

            embedder.encode(
                [sentence_a, sentence_b]
            )
        """

        print()
        print(
            "Semantic Redundancy Removal"
        )

        cleaned_groups: List[Dict] = []

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

            keep: List[str] = []

            keep_indices: List[int] = []

            for sentence in sentences:

                current_index = (
                    self.embedding_lookup.get(
                        sentence
                    )
                )

                duplicate = False

                if (
                    current_index is not None
                    and self.embedding_matrix
                    is not None
                ):

                    for existing_index in (
                        keep_indices
                    ):

                        similarity = (
                            torch.dot(
                                self.embedding_matrix[
                                    current_index
                                ],
                                self.embedding_matrix[
                                    existing_index
                                ],
                            ).item()
                        )

                        similarity = (
                            _clamp_similarity(
                                similarity
                            )
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

                    if (
                        current_index
                        is not None
                    ):
                        keep_indices.append(
                            current_index
                        )

            # --------------------------------------------------
            # Safety:
            #
            # Never allow a group to become empty because
            # of an embedding lookup issue.
            # --------------------------------------------------

            if not keep and sentences:

                keep = [
                    sentences[0]
                ]

            group[
                "sentences"
            ] = keep

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
    # STEP 4
    # ORDER SENTENCES
    # ======================================================

    def _order_sentences(
        self,
        groups: List[Dict],
        graph: List[Dict],
    ) -> List[str]:
        """
        Orders evidence using graph importance.

        Priority:

            1. Connectivity
            2. Semantic similarity
            3. Entity richness
        """

        print()
        print(
            "Graph-aware Ordering"
        )

        graph_lookup = {
            node["text"]: node
            for node in graph
        }

        scored_groups: List[
            Tuple[float, Dict]
        ] = []

        for group in groups:

            connectivity = 0.0

            for sentence in group[
                "sentences"
            ]:

                node = graph_lookup.get(
                    sentence
                )

                if node:

                    connectivity += (
                        self._graph_centrality(
                            node[
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

            # --------------------------------------------------
            # Normalize entity contribution.
            # --------------------------------------------------

            entity_score = min(
                len(
                    group[
                        "entities"
                    ]
                )
                / 5.0,
                1.0,
            )

            score = (
                0.45 * centrality
                + 0.35
                * group[
                    "similarity"
                ]
                + 0.20 * entity_score
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

        ordered: List[str] = []

        visited = set()

        for _, group in (
            scored_groups
        ):

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
    # STEP 5
    # BUILD FINAL PARAGRAPH
    # ======================================================

    def _build_paragraph(
        self,
        ordered_sentences: List[str],
    ) -> str:
        """
        Builds one coherent paragraph.

        Only performs whitespace and punctuation
        cleanup.

        It does NOT generate new facts.
        """

        print()
        print(
            "Paragraph Builder"
        )

        paragraph = " ".join(
            sentence.strip()
            for sentence in ordered_sentences
            if sentence.strip()
        )

        # Conservative whitespace cleanup.
        paragraph = " ".join(
            paragraph.split()
        )

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
        """
        Debug helper.

        Prints the complete sentence graph.
        """

        print()
        print(
            "=" * 60
        )

        print(
            "Evidence Graph"
        )

        print(
            "=" * 60
        )

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
                f"Entities : "
                f"{node['entities']}"
            )

            print(
                f"Neighbors : "
                f"{node['neighbors']}"
            )

            print(
                node["text"]
            )

        print()
        print(
            "=" * 60
        )

    # ======================================================
    # ENTITY OVERLAP
    # ======================================================

    def _entity_overlap(
        self,
        entities_a: List[str],
        entities_b: List[str],
    ) -> float:
        """
        Computes Jaccard overlap between
        two entity sets.
        """

        set_a = set(
            entities_a
        )

        set_b = set(
            entities_b
        )

        if (
            not set_a
            and not set_b
        ):
            return 0.0

        union = (
            set_a
            | set_b
        )

        if not union:
            return 0.0

        return (
            len(
                set_a
                & set_b
            )
            / len(union)
        )

    # ======================================================
    # SEMANTIC SIMILARITY
    # ======================================================

    def _semantic_similarity(
        self,
        sentence_a: str,
        sentence_b: str,
    ) -> float:
        """
        Return semantic similarity between two sentences.

        Optimization #18:
            This method NO LONGER performs model inference.

        It performs a lookup into the reusable
        embedding matrix.

        This method is retained for compatibility with
        existing code that may call it directly.
        """

        if (
            not sentence_a
            or not sentence_b
        ):
            return 0.0

        index_a = (
            self.embedding_lookup.get(
                sentence_a
            )
        )

        index_b = (
            self.embedding_lookup.get(
                sentence_b
            )
        )

        if (
            index_a is None
            or index_b is None
            or self.embedding_matrix
            is None
        ):

            # --------------------------------------------------
            # Compatibility fallback.
            #
            # This should normally never execute in the
            # production fuse() path because the matrix is
            # prepared before graph construction.
            # --------------------------------------------------

            embeddings = self.embedder.encode(
                [
                    sentence_a,
                    sentence_b,
                ],
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            self.embedding_fallbacks += 1

            similarity = util.cos_sim(
                embeddings[0],
                embeddings[1],
            ).item()

            return _clamp_similarity(
                similarity
            )

        similarity = torch.dot(
            self.embedding_matrix[
                index_a
            ],
            self.embedding_matrix[
                index_b
            ],
        ).item()

        return _clamp_similarity(
            similarity
        )

    # ======================================================
    # EDGE WEIGHT
    # ======================================================

    def _edge_weight(
        self,
        entities_a: List[str],
        entities_b: List[str],
        sentence_a: str,
        sentence_b: str,
    ) -> float:
        """
        Combines:

            Entity overlap
            +
            Semantic similarity

        into one edge score.

        Optimization #18:
            semantic similarity is a cheap embedding lookup.
        """

        overlap = (
            self._entity_overlap(
                entities_a,
                entities_b,
            )
        )

        semantic = (
            self._semantic_similarity(
                sentence_a,
                sentence_b,
            )
        )

        return (
            0.6 * overlap
            + 0.4 * semantic
        )

    # ======================================================
    # GRAPH CENTRALITY
    # ======================================================

    def _graph_centrality(
        self,
        neighbors: List[Dict],
    ) -> float:
        """
        Computes weighted graph centrality
        from neighbor edges.
        """

        return sum(
            neighbor.get(
                "weight",
                0.0,
            )
            for neighbor in neighbors
        )