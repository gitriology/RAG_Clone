"""Benchmark the exact FlatIP and HNSW retrieval indexes for Optimization #27."""

from __future__ import annotations

import argparse
import time

import numpy as np

from backend.retrieval.index.build_index import build_index, get_metadata
from backend.retrieval.utils.load_embeddings import load_embeddings


def benchmark(index_type: str, embeddings: np.ndarray, queries: np.ndarray, **kwargs):
    start = time.perf_counter()
    index = build_index(embeddings, index_type=index_type, **kwargs)
    build_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    index.search(np.ascontiguousarray(queries, dtype=np.float32), 10)
    search_ms = (time.perf_counter() - start) * 1000.0

    metadata = get_metadata(index)
    return metadata, build_ms, search_ms, search_ms / max(len(queries), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimization #27 FAISS benchmark")
    parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()

    embeddings = np.asarray(load_embeddings(), dtype=np.float32)
    query_count = min(max(args.queries, 1), len(embeddings))
    queries = embeddings[:query_count]

    print("\n" + "=" * 68)
    print("OPTIMIZATION #27 - FAISS INDEX BENCHMARK")
    print("=" * 68)
    print(f"Embeddings: {embeddings.shape[0]} x {embeddings.shape[1]}")
    print(f"Queries:    {query_count}")

    for index_type in ("flat", "hnsw"):
        metadata, build_ms, total_ms, per_query_ms = benchmark(
            index_type,
            embeddings,
            queries,
        )
        print(f"\n{index_type.upper()}")
        print(f"  metadata:       {metadata}")
        print(f"  build time ms:  {build_ms:.3f}")
        print(f"  search total ms: {total_ms:.3f}")
        print(f"  search/query ms: {per_query_ms:.3f}")

    print("\nUse this benchmark to justify the production index choice; #27 does not replace the index automatically.")


if __name__ == "__main__":
    main()
