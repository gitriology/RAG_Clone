"""
Phase 10.2
Evidence Fusion

Receives semantically selected evidence from Phase 10.1
and converts it into a structured evidence graph before
building a coherent paragraph.

Pipeline

Selected Sentences
        │
        ▼
Sentence Graph
        │
        ▼
Entity Merge
        │
        ▼
Redundancy Removal
        │
        ▼
Sentence Ordering
        │
        ▼
Evidence Paragraph
"""

from typing import List, Dict
from collections import defaultdict

import spacy

from sentence_transformers import util
import torch

from backend.retrieval.dense.embedder import model
# ----------------------------------------------------------
# spaCy model
# ----------------------------------------------------------

nlp = spacy.load("en_core_web_sm")


class EvidenceFusion:
    """
    Phase 10.2

    Evidence Fusion Engine.

    Input
    -----
    generate_answer()

    {
        "answer": "...",

        "selected_sentences":[
            {
                "rank":1,
                "text":"...",
                "similarity":0.91
            }
        ]
    }

    Output
    ------
    {

        "paragraph":"...",

        "graph":...,

        "entities":...,

        "sentence_count":...

    }
    """

    # ======================================================
    # Constructor
    # ======================================================

    def __init__(self):

        self.graph = defaultdict(list)

        self.entities = defaultdict(list)

        self.embedder = model

    # ======================================================
    # PUBLIC ENTRY
    # ======================================================

    def fuse(
        self,
        generation_result: Dict,
    ) -> Dict:
        """
        Main entry point.

        Executes the complete evidence fusion pipeline.
        """

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

        # ---------------------------------------------
        # Step 1
        # ---------------------------------------------

        graph = self._build_sentence_graph(
            selected
        )

        groups = self._merge_entities(
            graph
        )

        groups = self._remove_redundancy(
            groups
        )

        ordered = self._order_sentences(
            groups,
            graph,
        )

        paragraph = self._build_paragraph(
            ordered
        )

        print()

        print(
            f"Graph Nodes : {len(graph)}"
        )

        print("=" * 70)

        return {

            "paragraph": paragraph,

            "graph": graph,

            "groups": groups,

            "entities": dict(self.entities),

            "sentence_count": len(ordered),

        }

    # ======================================================
    # STEP 1
    # Sentence Graph Construction
    # ======================================================

    def _build_sentence_graph(
        self,
        selected_sentences: List[Dict],
    ) -> List[Dict]:
        """
        Builds an initial graph representation.

        Every sentence becomes one node.

        Nodes contain

        • sentence
        • similarity
        • named entities
        • outgoing links

        Edges are added whenever two sentences
        share at least one named entity.
        """

        nodes = []

        # ---------------------------------------------
        # Create nodes
        # ---------------------------------------------

        for idx, item in enumerate(
            selected_sentences
        ):

            sentence = item["text"]

            similarity = item.get(
                "similarity",
                0.0,
            )

            doc = nlp(sentence)

            entity_names = []

            for ent in doc.ents:

                name = ent.text.strip()

                entity_names.append(name)

                self.entities[
                    name
                ].append(idx)

            nodes.append(

                {

                    "id": idx,

                    "text": sentence,

                    "similarity": similarity,

                    "entities": entity_names,

                    "neighbors": [],

                }

            )

        # ---------------------------------------------
        # Connect nodes
        # ---------------------------------------------

        edge_threshold = 0.45

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

                weight = self._edge_weight(
                    list(entities_i),
                    list(entities_j),
                    nodes[i]["text"],
                    nodes[j]["text"],
                )

                if weight > edge_threshold:

                    nodes[i][
                        "neighbors"
                    ].append(
                        {
                            "id": j,
                            "weight": weight,
                        }
                    )

                    nodes[j][
                        "neighbors"
                    ].append(
                        {
                            "id": i,
                            "weight": weight,
                        }
                    )

        # ---------------------------------------------
        # Debug
        # ---------------------------------------------

        print()

        print("Sentence Graph")

        print(
            f"Nodes : {len(nodes)}"
        )

        print(
            f"Entities : {len(self.entities)}"
        )

        edge_count = sum(

            len(node["neighbors"])

            for node in nodes

        ) // 2

        print(
            f"Edges : {edge_count}"
        )

        return nodes
        # ======================================================
    # STEP 2
    # Merge Entity Groups
    # ======================================================

    def _merge_entities(
        self,
        nodes: List[Dict],
    ) -> List[Dict]:
        """
        Groups together sentences that discuss the
        same named entities.

        Instead of physically merging text, this stage
        enriches each node with a list of related
        sentences. The paragraph builder will later
        use these relationships to produce a coherent
        summary.
        """

        print()

        print("Entity Merge")

        merged = []

        visited = set()

        for node in nodes:

            if node["id"] in visited:
                continue

            group = [node]

            visited.add(node["id"])

            # -----------------------------------------
            # Find all directly connected neighbours
            # -----------------------------------------

            for neighbor in node["neighbors"]:

                neighbor_id = neighbor["id"]

                if neighbor["weight"] < 0.45:
                    continue

                if neighbor_id in visited:
                    continue

                group.append(nodes[neighbor_id])

                visited.add(neighbor_id)

            # -----------------------------------------
            # Collect unique entities
            # -----------------------------------------

            entity_set = set()

            for item in group:

                entity_set.update(
                    item["entities"]
                )

            # -----------------------------------------
            # Store grouped node
            # -----------------------------------------

            merged.append(

                {

                    "group_id": len(merged),

                    "sentences": [

                        g["text"]

                        for g in group

                    ],

                    "entities": sorted(
                        entity_set
                    ),

                    "similarity": max(

                        g["similarity"]

                        for g in group

                    ),

                }

            )

        print(
            f"Groups Created : {len(merged)}"
        )

        return merged

    # ======================================================
    # STEP 3
    # Remove Redundant Evidence
    # ======================================================

    def _remove_redundancy(
    self,
    groups: List[Dict],
    similarity_threshold: float = 0.92,
) -> List[Dict]:
        """
        Removes semantically redundant evidence using
        BGE embeddings.

        Sentences that convey nearly identical meaning
        are removed while preserving the highest-ranked
        evidence.
        """

        print()

        print("Semantic Redundancy Removal")

        cleaned_groups = []

        removed = 0

        for group in groups:

            sentences = group["sentences"]

            if len(sentences) <= 1:

                cleaned_groups.append(group)

                continue

            embeddings = model.encode(
                sentences,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )

            keep = []

            keep_embeddings = []

            for sentence, embedding in zip(
                sentences,
                embeddings,
            ):

                duplicate = False

                for existing in keep_embeddings:

                    similarity = util.cos_sim(
                        embedding,
                        existing,
                    ).item()

                    if similarity >= similarity_threshold:

                        duplicate = True

                        removed += 1

                        break

                if not duplicate:

                    keep.append(sentence)

                    keep_embeddings.append(embedding)

            group["sentences"] = keep

            cleaned_groups.append(group)

        print(f"Removed : {removed}")

        print(
            f"Remaining Groups : {len(cleaned_groups)}"
        )

        return cleaned_groups
        # ======================================================
    # STEP 4
    # Order Sentences
    # ======================================================

    def _order_sentences(
    self,
    groups: List[Dict],
    graph: List[Dict],
) -> List[str]:
        """
        Orders evidence using graph importance.

        Priority

        1. Connectivity
        2. Similarity
        3. Entity richness
        """

        print()

        print("Graph-aware Ordering")

        graph_lookup = {

            node["text"]: node

            for node in graph

        }

        scored_groups = []

        for group in groups:

            connectivity = 0.0

            for sentence in group["sentences"]:

                node = graph_lookup.get(sentence)

                if node:

                    connectivity += self._graph_centrality(
                        node["neighbors"]
                    )

            centrality = connectivity / max(
                len(group["sentences"]),
                1,
            )

            score = (

                0.45 * centrality

                +

                0.35 * group["similarity"]

                +

                0.20 * len(group["entities"])

            )

            scored_groups.append(

                (

                    score,

                    group,

                )

            )

        scored_groups.sort(

            key=lambda x: x[0],

            reverse=True,

        )

        ordered = []

        visited = set()

        for _, group in scored_groups:

            for sentence in group["sentences"]:

                if sentence in visited:

                    continue

                ordered.append(sentence)

                visited.add(sentence)

        print()

        print(

            f"Ordered {len(ordered)} evidence sentences"

        )

        return ordered

    # ======================================================
    # STEP 5
    # Build Final Paragraph
    # ======================================================

    def _build_paragraph(
        self,
        ordered_sentences: List[str],
    ) -> str:
        """
        Builds one coherent paragraph from
        ordered evidence.

        Performs small cleanup without
        modifying factual content.
        """

        print()

        print("Paragraph Builder")

        paragraph = " ".join(

            sentence.strip()

            for sentence in ordered_sentences

        )

        paragraph = paragraph.replace("  ", " ")

        if paragraph and not paragraph.endswith("."):

            paragraph += "."

        print(

            f"Paragraph Length : {len(paragraph.split())} words"

        )

        return paragraph

    # ======================================================
    # Helper
    # ======================================================

    def print_graph(
        self,
        graph: List[Dict],
    ):
        """
        Debug helper.

        Prints the sentence graph.
        """

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

                f"Similarity : {node['similarity']:.4f}"

            )

            print(

                f"Entities   : {node['entities']}"

            )

            print(

                f"Neighbors  : {node['neighbors']}"

            )

            print(

                node["text"]

            )

        print()

        print("=" * 60)

    def _entity_overlap(
        self,
        entities_a: List[str],
        entities_b: List[str],
    ) -> float:
        """
        Computes Jaccard overlap between two entity sets.
        """
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
        """
        Computes semantic similarity between two sentences
        using the existing BGE embedder.
        """
        if not sentence_a or not sentence_b:
            return 0.0

        embeddings = self.embedder.encode(
            [sentence_a, sentence_b],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        similarity = util.cos_sim(
            embeddings[0],
            embeddings[1],
        ).item()

        return max(0.0, min(1.0, similarity))

    def _edge_weight(
        self,
        entities_a: List[str],
        entities_b: List[str],
        sentence_a: str,
        sentence_b: str,
    ) -> float:
        """
        Combines entity overlap and semantic similarity
        into a single weighted edge score.
        """
        overlap = self._entity_overlap(entities_a, entities_b)
        semantic = self._semantic_similarity(sentence_a, sentence_b)

        return (0.6 * overlap) + (0.4 * semantic)

    def _graph_centrality(
        self,
        neighbors: List[Dict],
    ) -> float:
        """
        Computes weighted graph centrality from neighbor edges.
        """
        return sum(
            neighbor.get("weight", 0.0)
            for neighbor in neighbors
        )