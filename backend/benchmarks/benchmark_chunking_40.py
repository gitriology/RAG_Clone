"""
Optimization #40 — experimental chunk-size/overlap evaluation.

The paper requirement is explicit: compare 200/40, 300/60, 400/80,
and 500/100 rather than assuming a fixed chunk size is optimal.

This benchmark uses the project's labelled retrieval queries
(`backend/evaluation/fusion_dataset.json`) and the existing processed
corpus. The current processed chunks are reconstructed approximately
per source document (removing detected sentence/token overlap), then
re-chunked with each candidate configuration.

Metrics:
    - Recall@5: relevant source document appears in top-5 chunks.
    - Precision@5: fraction of top-5 chunks whose source is relevant.
    - MRR@5: reciprocal rank of first relevant chunk.
    - Context relevance: query-token coverage averaged over top-5.
    - Answer-support accuracy: whether top-3 context covers >=50% of the
      labelled answer tokens. This is an extractive support proxy, not
      an LLM factuality claim.
    - Latency: chunking + index build + retrieval time for the benchmark.

No production default is changed by this benchmark. A winner should only
be adopted after the measured results justify it.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from backend.data_pipeline.chunking.chunker import chunk_text


BASE_DIR = Path(__file__).resolve().parents[2]
CORPUS_FILE = BASE_DIR / "data" / "processed" / "all_domains.json"
EVAL_FILE = BASE_DIR / "backend" / "evaluation" / "fusion_dataset.json"

CONFIGS: Tuple[Tuple[int, int], ...] = (
    (200, 40),
    (300, 60),
    (400, 80),
    (500, 100),
)

TOP_K = 5
ANSWER_TOP_K = 3
ANSWER_COVERAGE_THRESHOLD = 0.50
OVERLAP_MIN_TOKENS = 5

TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def token_set(text: str) -> Set[str]:
    return set(tokenize(text))


def longest_suffix_prefix_overlap(left: Sequence[str], right: Sequence[str]) -> int:
    max_len = min(len(left), len(right))
    for size in range(max_len, OVERLAP_MIN_TOKENS - 1, -1):
        if list(left[-size:]) == list(right[:size]):
            return size
    return 0


def reconstruct_documents(records: Sequence[dict]) -> Dict[str, dict]:
    """
    Reconstruct approximate document text from the current processed chunks.

    This is needed because the ZIP intentionally contains the processed
    evaluation corpus but not every original PDF. Existing chunk overlap is
    removed before the experimental re-chunking step.
    """
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record["document_id"])].append(record)

    documents = {}
    for document_id, items in grouped.items():
        items.sort(key=lambda x: int(x.get("chunk_index", 0)))
        tokens: List[str] = []
        for item in items:
            current = tokenize(item.get("text", ""))
            if not current:
                continue
            overlap = longest_suffix_prefix_overlap(tokens, current)
            tokens.extend(current[overlap:])
        text = " ".join(tokens).strip()
        first = items[0]
        documents[document_id] = {
            "document_id": document_id,
            "domain": first.get("domain", "general"),
            "source": first.get("source", "unknown"),
            "text": text,
        }
    return documents


def load_evaluation(records: Sequence[dict]) -> List[dict]:
    with EVAL_FILE.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    id_to_document = {
        str(record["id"]): str(record["document_id"])
        for record in records
    }

    evaluation = []
    for item in raw:
        relevant_sources = {
            id_to_document[str(doc_id)]
            for doc_id in item.get("relevant_doc_ids", [])
            if str(doc_id) in id_to_document
        }
        if not relevant_sources:
            continue

        gold_texts = []
        for relevant_id in item.get("relevant_doc_ids", []):
            for candidate in item.get("candidates", []):
                if str(candidate.get("doc_id")) == str(relevant_id):
                    gold_texts.append(candidate.get("text", ""))
                    break

        evaluation.append(
            {
                "query": item["query"],
                "relevant_sources": relevant_sources,
                "gold_texts": gold_texts,
            }
        )
    return evaluation


def bm25_idf(doc_freq: int, total_docs: int) -> float:
    # Standard BM25 IDF with a +1 floor.
    return max(
        0.0,
        ((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0),
    )


class SimpleBM25:
    """Small dependency-free BM25 implementation for the benchmark."""

    def __init__(self, texts: Sequence[str]):
        self.texts = list(texts)
        self.tokens = [tokenize(text) for text in self.texts]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.avgdl = statistics.mean(self.lengths) if self.lengths else 0.0
        self.doc_freq = Counter()
        for tokens in self.tokens:
            for term in set(tokens):
                self.doc_freq[term] += 1

    def scores(self, query: str) -> List[float]:
        q_tokens = tokenize(query)
        total_docs = len(self.tokens)
        if total_docs == 0:
            return []

        k1 = 1.5
        b = 0.75
        query_counts = Counter(q_tokens)
        scores = []

        for tokens, length in zip(self.tokens, self.lengths):
            counts = Counter(tokens)
            score = 0.0
            denominator_length = max(self.avgdl, 1.0)
            for term, qtf in query_counts.items():
                tf = counts.get(term, 0)
                if not tf:
                    continue
                idf = bm25_idf(self.doc_freq.get(term, 0), total_docs)
                denominator = tf + k1 * (
                    1.0 - b + b * length / denominator_length
                )
                score += idf * (tf * (k1 + 1.0) / denominator)
            scores.append(score)
        return scores

    def top_indices(self, query: str, top_k: int) -> List[int]:
        scores = self.scores(query)
        return sorted(
            range(len(scores)),
            key=lambda index: (-scores[index], index),
        )[:top_k]


def make_variant_chunks(documents: Dict[str, dict], chunk_size: int, overlap: int) -> List[dict]:
    chunks = []
    for document_id, document in documents.items():
        parts = chunk_text(
            document["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )
        for chunk_index, text in enumerate(parts):
            chunks.append(
                {
                    "document_id": document_id,
                    "source": document["source"],
                    "domain": document["domain"],
                    "chunk_index": chunk_index,
                    "text": text,
                }
            )
    return chunks


def context_relevance(query: str, texts: Sequence[str]) -> float:
    q = token_set(query)
    if not q:
        return 0.0
    if not texts:
        return 0.0
    return statistics.mean(
        len(q.intersection(token_set(text))) / len(q)
        for text in texts
    )


def answer_support_accuracy(
    evaluation_item: dict,
    retrieved_texts: Sequence[str],
) -> float:
    """
    Return 1 when the retrieved top-3 context contains >=50% of the
    labelled answer tokens. Multiple gold texts are combined.
    """
    gold = " ".join(evaluation_item.get("gold_texts", []))
    gold_tokens = token_set(gold)
    if not gold_tokens:
        return 0.0

    context_tokens = token_set(" ".join(retrieved_texts[:ANSWER_TOP_K]))
    coverage = len(gold_tokens.intersection(context_tokens)) / len(gold_tokens)
    return 1.0 if coverage >= ANSWER_COVERAGE_THRESHOLD else 0.0


def evaluate_variant(chunks: Sequence[dict], evaluation: Sequence[dict]) -> dict:
    texts = [chunk["text"] for chunk in chunks]
    source_ids = [chunk["document_id"] for chunk in chunks]

    start = time.perf_counter()
    index = SimpleBM25(texts)
    index_build_seconds = time.perf_counter() - start

    query_start = time.perf_counter()
    recalls = []
    precisions = []
    reciprocal_ranks = []
    context_scores = []
    answer_support = []

    for item in evaluation:
        indices = index.top_indices(item["query"], TOP_K)
        retrieved_sources = [source_ids[i] for i in indices]
        retrieved_texts = [texts[i] for i in indices]

        relevant = item["relevant_sources"]
        hits = [source in relevant for source in retrieved_sources]

        recalls.append(1.0 if any(hits) else 0.0)
        precisions.append(sum(hits) / TOP_K if TOP_K else 0.0)

        rr = 0.0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        context_scores.append(
            context_relevance(item["query"], retrieved_texts)
        )
        answer_support.append(
            answer_support_accuracy(item, retrieved_texts)
        )

    query_seconds = time.perf_counter() - query_start

    return {
        "chunk_count": len(chunks),
        "index_build_ms": index_build_seconds * 1000.0,
        "retrieval_ms": query_seconds * 1000.0,
        "total_ms": (index_build_seconds + query_seconds) * 1000.0,
        "recall_at_5": statistics.mean(recalls),
        "precision_at_5": statistics.mean(precisions),
        "mrr_at_5": statistics.mean(reciprocal_ranks),
        "context_relevance": statistics.mean(context_scores),
        "answer_support_accuracy": statistics.mean(answer_support),
    }


def main() -> None:
    with CORPUS_FILE.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    documents = reconstruct_documents(records)
    evaluation = load_evaluation(records)

    if not evaluation:
        raise RuntimeError("No labelled evaluation queries are available.")

    print("=" * 78)
    print("OPTIMIZATION #40 — CHUNK SIZE / OVERLAP EXPERIMENT")
    print("=" * 78)
    print(f"Source records:       {len(records)}")
    print(f"Reconstructed docs:   {len(documents)}")
    print(f"Evaluation queries:   {len(evaluation)}")
    print(f"Configurations:       {', '.join(f'{s}/{o}' for s, o in CONFIGS)}")
    print()
    print(
        "Metrics: Recall@5, Precision@5, MRR@5, context relevance, "
        "answer-support accuracy, latency."
    )

    results = []
    for chunk_size, overlap in CONFIGS:
        start = time.perf_counter()
        chunks = make_variant_chunks(documents, chunk_size, overlap)
        chunking_seconds = time.perf_counter() - start

        result = evaluate_variant(chunks, evaluation)
        result.update(
            {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "chunking_ms": chunking_seconds * 1000.0,
                "end_to_end_benchmark_ms":
                    chunking_seconds * 1000.0 + result["total_ms"],
            }
        )
        results.append(result)

        print(f"\n{chunk_size}/{overlap}")
        print(f"  chunks:                    {result['chunk_count']}")
        print(f"  Recall@5:                  {result['recall_at_5']:.3f}")
        print(f"  Precision@5:               {result['precision_at_5']:.3f}")
        print(f"  MRR@5:                     {result['mrr_at_5']:.3f}")
        print(f"  Context relevance:         {result['context_relevance']:.3f}")
        print(f"  Answer-support accuracy:   {result['answer_support_accuracy']:.3f}")
        print(f"  Chunking time:             {result['chunking_ms']:.3f} ms")
        print(f"  Index build time:          {result['index_build_ms']:.3f} ms")
        print(f"  Retrieval time:            {result['retrieval_ms']:.3f} ms")
        print(f"  Total benchmark time:      {result['end_to_end_benchmark_ms']:.3f} ms")

    # Primary selection: maximize answer-support accuracy, then recall,
    # then context relevance, then minimize latency.
    winner = max(
        results,
        key=lambda r: (
            r["answer_support_accuracy"],
            r["recall_at_5"],
            r["context_relevance"],
            -r["end_to_end_benchmark_ms"],
        ),
    )

    print("\n" + "-" * 78)
    print(
        "Measured winner (benchmark tie-break policy): "
        f"{winner['chunk_size']}/{winner['overlap']}"
    )
    print(
        "This winner is evidence for the benchmark corpus only; "
        "production adoption should follow the same configuration "
        "after a broader labelled evaluation set."
    )

    output = BASE_DIR / "backend" / "benchmarks" / "chunking_40_results.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "optimization": 40,
                "evaluation_queries": len(evaluation),
                "configurations": [
                    {"chunk_size": s, "overlap": o}
                    for s, o in CONFIGS
                ],
                "winner": {
                    "chunk_size": winner["chunk_size"],
                    "overlap": winner["overlap"],
                },
                "results": results,
                "notes": [
                    "Corpus text is reconstructed from existing processed chunks because the ZIP does not contain all original PDFs.",
                    "Answer-support accuracy is an extractive labelled-context proxy, not factual entailment.",
                    "Latency is benchmark retrieval latency using dependency-free BM25; it is not end-to-end production RAG latency.",
                ],
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
