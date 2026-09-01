"""
Three-way diagnostic benchmark for the MS-ARC project.

IMPORTANT:
    This is a STANDALONE evaluation script.
    It does NOT modify any existing project source files.

Purpose:
    Run the SAME five already-labelled evaluation queries through:

        1. Fixed-K baseline
        2. Complexity-Adaptive baseline
        3. MS-ARC

    and save a complete JSON record containing:
        - query
        - ground truth
        - selected K
        - candidate pool depth
        - retrieved document IDs
        - top-3 retrieved IDs
        - Recall@3
        - Precision@3
        - HitRate@3
        - MRR@3
        - complexity signals
        - MS-ARC agreement/margin/stability
        - MS-ARC confidence/decision
        - reproducibility configuration

The five queries are read from the project's existing:
    backend/evaluation/fusion_dataset.json

That file already contains manually verified relevant_doc_ids.
The script does NOT infer ground truth from retrieval results.

Run from the PROJECT ROOT:
    python three_way_diagnostic.py

Or:
    python /path/to/three_way_diagnostic.py

The output JSON is written OUTSIDE the project by default:
    ./three_way_diagnostic_results.json

If --output is supplied, that exact path is used.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Fixed vs Complexity-Adaptive vs MS-ARC diagnostics."
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help=(
            "Path to the RAG-optimisation project root. "
            "Defaults to the directory containing backend/."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output JSON path. "
            "Defaults to <project-root>/../three_way_diagnostic_results.json "
            "so no project source/evaluation file is overwritten."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of labelled queries to evaluate. Default: 5.",
    )

    parser.add_argument(
        "--fixed-k",
        type=int,
        default=3,
        help="Fixed baseline retrieval depth. Default: 3.",
    )

    parser.add_argument(
        "--metric-k",
        type=int,
        default=3,
        help="Evaluation cutoff for all methods. Default: 3.",
    )

    parser.add_argument(
        "--fusion",
        type=str,
        default="minmax",
        choices=("minmax", "rrf"),
        help="Hybrid fusion method. Default: minmax.",
    )

    return parser.parse_args()


# ============================================================
# PATH DISCOVERY
# ============================================================

def discover_project_root(explicit_root: str | None) -> Path:
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
    else:
        # If the script is copied to the project root, this is correct.
        here = Path(__file__).resolve()

        if (here / "backend").is_dir():
            root = here
        elif (here.parent / "backend").is_dir():
            root = here.parent
        else:
            root = Path.cwd().resolve()

    if not (root / "backend").is_dir():
        raise FileNotFoundError(
            "Could not find the project root.\n"
            "Expected a directory containing 'backend/'.\n"
            "Use --project-root explicitly."
        )

    return root


# ============================================================
# SAFE CONVERSION
# ============================================================

def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def json_safe(value: Any) -> Any:
    """
    Convert common NumPy/Python scalar containers into JSON-safe values.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float):
            if not math.isfinite(value):
                return None
        return value

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    # NumPy scalar / array support without requiring NumPy explicitly.
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass

    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass

    return str(value)


# ============================================================
# DATASET
# ============================================================

def load_evaluation_dataset(project_root: Path) -> List[Dict[str, Any]]:
    path = (
        project_root
        / "backend"
        / "evaluation"
        / "fusion_dataset.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found:\n{path}"
        )

    with path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError(
            f"Expected a list in {path}, got {type(dataset).__name__}."
        )

    labelled = []

    for item in dataset:
        if not isinstance(item, dict):
            continue

        query = item.get("query")
        relevant = item.get("relevant_doc_ids")

        # Only use manually-labelled entries.
        if (
            isinstance(query, str)
            and query.strip()
            and isinstance(relevant, list)
            and len(relevant) > 0
        ):
            labelled.append(item)

    return labelled


def load_all_domains(project_root: Path) -> List[Dict[str, Any]]:
    path = (
        project_root
        / "data"
        / "processed"
        / "all_domains.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"all_domains.json not found:\n{path}"
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a list in {path}, got {type(data).__name__}."
        )

    return data


def validate_ground_truth(
    evaluation_rows: Sequence[Dict[str, Any]],
    all_domains: Sequence[Dict[str, Any]],
) -> None:
    """
    Validate that each ground-truth ID is a valid retrieval row index.

    The project's retrieval layer uses FAISS row indices as doc_id values.
    We therefore evaluate against row indices, not merely the dataset's
    human-readable 'id' field.
    """
    n = len(all_domains)

    for row in evaluation_rows:
        query = row["query"]

        for raw_id in row["relevant_doc_ids"]:
            doc_id = as_int(raw_id, default=-1)

            if doc_id < 0 or doc_id >= n:
                raise ValueError(
                    f"Invalid ground-truth retrieval row {raw_id!r} "
                    f"for query {query!r}. all_domains has {n} rows."
                )


# ============================================================
# METRICS
# ============================================================

def normalize_ids(values: Iterable[Any]) -> List[str]:
    result = []

    for value in values:
        if value is None:
            continue
        result.append(str(value))

    return result


def metric_values(
    retrieved_ids: Sequence[Any],
    relevant_ids: Sequence[Any],
    k: int,
) -> Dict[str, float]:
    retrieved = normalize_ids(retrieved_ids[:k])
    relevant = set(normalize_ids(relevant_ids))

    if not relevant:
        return {
            "recall_at_3": 0.0,
            "precision_at_3": 0.0,
            "hit_rate_at_3": 0.0,
            "mrr_at_3": 0.0,
        }

    retrieved_set = set(retrieved)
    hits = len(retrieved_set & relevant)

    recall = hits / len(relevant)
    precision = hits / len(retrieved) if retrieved else 0.0
    hit_rate = 1.0 if hits > 0 else 0.0

    reciprocal_rank = 0.0

    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "recall_at_3": recall,
        "precision_at_3": precision,
        "hit_rate_at_3": hit_rate,
        "mrr_at_3": reciprocal_rank,
    }


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def document_id(document: Any) -> str:
    if isinstance(document, dict):
        value = document.get("doc_id")
        if value is None:
            value = document.get("id")
    else:
        value = getattr(document, "doc_id", None)
        if value is None:
            value = getattr(document, "id", None)

    if value is None:
        raise ValueError(
            f"Could not extract doc_id from document: {document!r}"
        )

    return str(value)


def document_summary(
    document: Any,
    rank: int,
) -> Dict[str, Any]:
    """
    Preserve useful retrieval information without storing full document text.
    """
    if isinstance(document, dict):
        metadata = document.get("metadata", {}) or {}

        return {
            "rank": rank,
            "doc_id": str(
                document.get(
                    "doc_id",
                    document.get("id"),
                )
            ),
            "dense_score": as_float(
                document.get("dense_score", 0.0)
            ),
            "sparse_score": as_float(
                document.get(
                    "sparse_score",
                    document.get("bm25_score", 0.0),
                )
            ),
            "hybrid_score": as_float(
                document.get("hybrid_score", 0.0)
            ),
            "rerank_score": (
                as_float(document["rerank_score"])
                if document.get("rerank_score") is not None
                else None
            ),
            "dataset_id": metadata.get("dataset_id"),
            "domain": metadata.get("domain"),
            "source": metadata.get("source"),
            "faiss_row": metadata.get("faiss_row"),
        }

    metadata = getattr(document, "metadata", {}) or {}

    rerank_score = getattr(document, "rerank_score", None)

    return {
        "rank": rank,
        "doc_id": str(getattr(document, "doc_id")),
        "dense_score": as_float(
            getattr(document, "dense_score", 0.0)
        ),
        "sparse_score": as_float(
            getattr(document, "sparse_score", 0.0)
        ),
        "hybrid_score": as_float(
            metadata.get("hybrid_score", 0.0)
        ),
        "rerank_score": (
            as_float(rerank_score)
            if rerank_score is not None
            else None
        ),
        "dataset_id": metadata.get("dataset_id"),
        "domain": metadata.get("domain"),
        "source": metadata.get("source"),
        "faiss_row": metadata.get("faiss_row"),
    }


# ============================================================
# COMPLEXITY
# ============================================================

def run_complexity_analysis(query: str):
    from backend.ms_arc.complexity.analyzer import (
        QueryComplexityAnalyzer,
    )
    from backend.ms_arc.state.retrieval_state import RetrievalState

    analyzer = QueryComplexityAnalyzer()

    state = RetrievalState(query=query)
    state = analyzer.analyze(state)

    return state


def complexity_record(state: Any) -> Dict[str, Any]:
    return {
        "complexity": as_float(
            getattr(state, "query_complexity", 0.0)
        ),
        "query_type": getattr(
            state,
            "query_type",
            "",
        ),
        "recommended_topk": as_int(
            getattr(state, "recommended_topk", 0)
        ),
        "complexity_details": json_safe(
            getattr(state, "debug", {}).get(
                "complexity",
                {},
            )
        ),
    }


# ============================================================
# BASELINE RETRIEVAL
# ============================================================

def run_hybrid(
    query: str,
    k: int,
    candidate_k: int | None,
    fusion_method: str,
    hybrid_search_fn,
    retrieval_module,
) -> Dict[str, Any]:
    results = hybrid_search_fn(
        query=query,
        faiss_index=retrieval_module.faiss_index,
        bm25=retrieval_module.bm25,
        texts=retrieval_module.texts,
        k=k,
        candidate_k=candidate_k,
        fusion_method=fusion_method,
    )

    merged = results.get("merged_results", [])

    return {
        "candidate_k_used": as_int(
            results.get("candidate_k"),
            default=0,
        ),
        "dense_results": [
            document_summary(doc, rank)
            for rank, doc in enumerate(
                results.get("dense_results", []),
                start=1,
            )
        ],
        "sparse_results": [
            document_summary(doc, rank)
            for rank, doc in enumerate(
                results.get("sparse_results", []),
                start=1,
            )
        ],
        "merged_results": [
            document_summary(doc, rank)
            for rank, doc in enumerate(
                merged,
                start=1,
            )
        ],
    }


# ============================================================
# MS-ARC
# ============================================================

def run_msarc_diagnostic(
    query: str,
    fusion_method: str,
) -> Dict[str, Any]:
    from backend.ms_arc.run_msarc import run_msarc

    state = run_msarc(
        query,
        fusion_method=fusion_method,
    )

    # MS-ARC's actual final ranking is the CrossEncoder-reranked list.
    final_documents = (
        state.reranked_results
        if state.reranked_results
        else state.selected_documents
    )

    decision = state.signals.decision
    agreement = state.signals.agreement
    margin = state.signals.margin
    stability = state.signals.stability

    debug = getattr(state, "debug", {}) or {}

    return {
        "complexity": as_float(
            state.query_complexity
        ),
        "query_type": state.query_type,
        "recommended_topk": as_int(
            state.recommended_topk
        ),

        "candidate_k_used": (
            as_int(
                debug.get("candidate_k_used"),
                default=0,
            )
            if debug.get("candidate_k_used") is not None
            else None
        ),

        "fusion_method": debug.get(
            "fusion_method",
            fusion_method,
        ),

        "agreement": {
            "score": as_float(
                agreement.score
            ),
            "intersection": as_int(
                agreement.intersection
            ),
            "union": as_int(
                agreement.union
            ),
        },

        "margin": {
            "top_score": as_float(
                margin.top_score
            ),
            "second_score": as_float(
                margin.second_score
            ),
            "raw_margin": as_float(
                margin.raw_margin
            ),
            "normalized_margin": as_float(
                margin.normalized_margin
            ),
        },

        "stability": {
            "score": as_float(
                stability.score
            ),
            "overlap_1": as_float(
                stability.overlap_1
            ),
            "overlap_2": as_float(
                stability.overlap_2
            ),
            "topk_runs": [
                as_int(x)
                for x in stability.topk_runs
            ],
        },

        "decision": {
            "confidence": as_float(
                decision.confidence
            ),
            "decision": decision.decision,
            "reason": decision.reason,
            "agreement_weight": as_float(
                decision.agreement_weight
            ),
            "margin_weight": as_float(
                decision.margin_weight
            ),
            "stability_weight": as_float(
                decision.stability_weight
            ),
        },

        "retrieved_documents": [
            document_summary(doc, rank)
            for rank, doc in enumerate(
                final_documents,
                start=1,
            )
        ],

        "selected_documents_before_reranking": [
            document_summary(doc, rank)
            for rank, doc in enumerate(
                state.selected_documents,
                start=1,
            )
        ],

        "dense_results": [
            document_summary(doc, rank)
            for rank, doc in enumerate(
                state.dense_results,
                start=1,
            )
        ],

        "sparse_results": [
            document_summary(doc, rank)
            for rank, doc in enumerate(
                state.sparse_results,
                start=1,
            )
        ],

        "merged_results_before_reranking": [
            document_summary(doc, rank)
            for rank, doc in enumerate(
                state.merged_results,
                start=1,
            )
        ],
    }


# ============================================================
# SINGLE QUERY
# ============================================================

def evaluate_query(
    row: Dict[str, Any],
    all_domains: Sequence[Dict[str, Any]],
    fixed_k: int,
    metric_k: int,
    fusion_method: str,
    hybrid_search_fn,
    retrieval_module,
) -> Dict[str, Any]:

    query = row["query"].strip()

    relevant_ids = [
        str(x)
        for x in row["relevant_doc_ids"]
    ]

    # --------------------------------------------------------
    # Verify GT rows and capture their metadata.
    # --------------------------------------------------------

    ground_truth_documents = []

    for raw_id in row["relevant_doc_ids"]:
        row_id = as_int(raw_id, default=-1)

        if row_id < 0 or row_id >= len(all_domains):
            raise ValueError(
                f"Ground-truth row {raw_id} is invalid for {query!r}."
            )

        doc = all_domains[row_id]

        ground_truth_documents.append({
            "retrieval_doc_id": str(row_id),
            "dataset_id": doc.get("id"),
            "domain": doc.get("domain"),
            "source": doc.get("source"),
            "source_path": doc.get("source_path"),
            "chunk_index": doc.get("chunk_index"),
            "page_start": doc.get("page_start"),
            "page_end": doc.get("page_end"),
        })

    # --------------------------------------------------------
    # Complexity analysis used by the adaptive baseline.
    # --------------------------------------------------------

    complexity_state = run_complexity_analysis(query)
    complexity = complexity_record(complexity_state)

    # --------------------------------------------------------
    # FIXED
    # --------------------------------------------------------

    fixed = run_hybrid(
        query=query,
        k=fixed_k,
        candidate_k=max(fixed_k * 4, 20),
        fusion_method=fusion_method,
        hybrid_search_fn=hybrid_search_fn,
        retrieval_module=retrieval_module,
    )

    fixed_full_ids = [
        item["doc_id"]
        for item in fixed["merged_results"]
    ]

    fixed_top_metric_ids = fixed_full_ids[:metric_k]

    fixed["k"] = fixed_k
    fixed["metric_k"] = metric_k
    fixed["retrieved_doc_ids"] = fixed_full_ids
    fixed["top_metric_doc_ids"] = fixed_top_metric_ids
    fixed["metrics"] = metric_values(
        fixed_top_metric_ids,
        relevant_ids,
        metric_k,
    )

    # --------------------------------------------------------
    # COMPLEXITY-ADAPTIVE
    # --------------------------------------------------------

    adaptive_k = int(
        complexity_state.recommended_topk
    )

    adaptive = run_hybrid(
        query=query,
        k=adaptive_k,
        candidate_k=None,
        fusion_method=fusion_method,
        hybrid_search_fn=hybrid_search_fn,
        retrieval_module=retrieval_module,
    )

    adaptive_full_ids = [
        item["doc_id"]
        for item in adaptive["merged_results"]
    ]

    adaptive_top_metric_ids = adaptive_full_ids[:metric_k]

    adaptive["complexity"] = complexity["complexity"]
    adaptive["query_type"] = complexity["query_type"]
    adaptive["recommended_topk"] = complexity["recommended_topk"]
    adaptive["k"] = adaptive_k
    adaptive["metric_k"] = metric_k
    adaptive["retrieved_doc_ids"] = adaptive_full_ids
    adaptive["top_metric_doc_ids"] = adaptive_top_metric_ids
    adaptive["metrics"] = metric_values(
        adaptive_top_metric_ids,
        relevant_ids,
        metric_k,
    )

    # --------------------------------------------------------
    # MS-ARC
    # --------------------------------------------------------

    msarc = run_msarc_diagnostic(
        query=query,
        fusion_method=fusion_method,
    )

    msarc_full_ids = [
        item["doc_id"]
        for item in msarc["retrieved_documents"]
    ]

    msarc_top_metric_ids = msarc_full_ids[:metric_k]

    msarc["metric_k"] = metric_k
    msarc["retrieved_doc_ids"] = msarc_full_ids
    msarc["top_metric_doc_ids"] = msarc_top_metric_ids
    msarc["metrics"] = metric_values(
        msarc_top_metric_ids,
        relevant_ids,
        metric_k,
    )

    # --------------------------------------------------------
    # Query record
    # --------------------------------------------------------

    return {
        "query": query,

        "ground_truth": {
            "relevant_retrieval_doc_ids": relevant_ids,
            "documents": ground_truth_documents,
        },

        "complexity_analysis": complexity,

        "fixed": fixed,

        "complexity_adaptive": adaptive,

        "ms_arc": msarc,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    args = parse_args()

    project_root = discover_project_root(
        args.project_root
    )

    if args.fixed_k <= 0:
        raise ValueError("--fixed-k must be > 0.")

    if args.metric_k <= 0:
        raise ValueError("--metric-k must be > 0.")

    if args.limit <= 0:
        raise ValueError("--limit must be > 0.")

    # --------------------------------------------------------
    # Import project retrieval machinery.
    #
    # IMPORTANT:
    # These are imported from the existing project.
    # This script does not edit them.
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("MS-ARC THREE-WAY DIAGNOSTIC BENCHMARK")
    print("=" * 80)
    print(f"Project root : {project_root}")
    print(f"Fixed K      : {args.fixed_k}")
    print(f"Metric K     : {args.metric_k}")
    print(f"Fusion       : {args.fusion}")
    print()

    dataset = load_evaluation_dataset(
        project_root
    )

    all_domains = load_all_domains(
        project_root
    )

    if len(dataset) < args.limit:
        raise ValueError(
            f"Only {len(dataset)} labelled queries were found in "
            f"fusion_dataset.json, but --limit={args.limit}."
        )

    #rows = dataset[:args.limit]
    rows = dataset[5:10]

    validate_ground_truth(
        rows,
        all_domains,
    )

    print(
        f"Using {len(rows)} labelled queries from "
        "backend/evaluation/fusion_dataset.json"
    )

    print(
        f"Corpus rows in all_domains.json: "
        f"{len(all_domains)}"
    )

    # --------------------------------------------------------
    # Existing retrieval components.
    # --------------------------------------------------------

    from backend.ms_arc.retrieval import retrieve as retrieval_module
    from backend.retrieval.hybrid.hybrid_search import hybrid_search

    results = []

    for index, row in enumerate(rows, start=1):

        print()
        print("=" * 80)
        print(
            f"QUERY {index}/{len(rows)}"
        )
        print(
            f"{row['query']}"
        )
        print(
            f"Ground truth: "
            f"{row['relevant_doc_ids']}"
        )
        print("=" * 80)

        try:
            result = evaluate_query(
                row=row,
                all_domains=all_domains,
                fixed_k=args.fixed_k,
                metric_k=args.metric_k,
                fusion_method=args.fusion,
                hybrid_search_fn=hybrid_search,
                retrieval_module=retrieval_module,
            )

            results.append(result)

            # Compact diagnostic output.
            print()
            print(
                "[Fixed]"
            )
            print(
                "  K:",
                result["fixed"]["k"],
            )
            print(
                "  Top-3:",
                result["fixed"]["top_metric_doc_ids"],
            )
            print(
                "  Metrics:",
                result["fixed"]["metrics"],
            )

            print()
            print(
                "[Complexity-Adaptive]"
            )
            print(
                "  Complexity:",
                result["complexity_analysis"]["complexity"],
            )
            print(
                "  Type:",
                result["complexity_analysis"]["query_type"],
            )
            print(
                "  K:",
                result["complexity_adaptive"]["k"],
            )
            print(
                "  Top-3:",
                result["complexity_adaptive"]["top_metric_doc_ids"],
            )
            print(
                "  Metrics:",
                result["complexity_adaptive"]["metrics"],
            )

            print()
            print(
                "[MS-ARC]"
            )
            print(
                "  Complexity:",
                result["ms_arc"]["complexity"],
            )
            print(
                "  Type:",
                result["ms_arc"]["query_type"],
            )
            print(
                "  Recommended K:",
                result["ms_arc"]["recommended_topk"],
            )
            print(
                "  Candidate K used:",
                result["ms_arc"]["candidate_k_used"],
            )
            print(
                "  Agreement:",
                result["ms_arc"]["agreement"]["score"],
            )
            print(
                "  Margin:",
                result["ms_arc"]["margin"]["normalized_margin"],
            )
            print(
                "  Stability:",
                result["ms_arc"]["stability"]["score"],
            )
            print(
                "  Confidence:",
                result["ms_arc"]["decision"]["confidence"],
            )
            print(
                "  Decision:",
                result["ms_arc"]["decision"]["decision"],
            )
            print(
                "  Top-3:",
                result["ms_arc"]["top_metric_doc_ids"],
            )
            print(
                "  Metrics:",
                result["ms_arc"]["metrics"],
            )

        except Exception as exc:
            print()
            print(
                "[ERROR] Query failed:"
            )
            print(
                f"{type(exc).__name__}: {exc}"
            )
            raise

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        # Deliberately outside backend/evaluation so the existing
        # evaluation files remain untouched.
        output_path = (
            project_root.parent
            / "three_way_diagnostic_results.json"
        )

    payload = {
        "experiment": {
            "name": "MS-ARC three-way diagnostic",
            "purpose": (
                "Diagnostic comparison of Fixed-K, "
                "Complexity-Adaptive, and MS-ARC."
            ),
            "queries_used": len(results),
            "query_source": (
                "backend/evaluation/fusion_dataset.json"
            ),
            "corpus_source": (
                "data/processed/all_domains.json"
            ),
            "ground_truth_source": (
                "manually-labelled relevant_doc_ids "
                "in fusion_dataset.json"
            ),
            "fixed_k": args.fixed_k,
            "metric_k": args.metric_k,
            "metrics": [
                "Recall@3",
                "Precision@3",
                "HitRate@3",
                "MRR@3",
            ],
            "fusion_method": args.fusion,
            "evaluation_rule": (
                "All three methods are evaluated using their "
                "first metric_k retrieved documents."
            ),
            "note": (
                "This diagnostic records the current implementation "
                "as-is. It does not modify or repair MS-ARC behavior."
            ),
        },
        "queries": results,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_safe(payload),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print(
        f"Saved JSON:\n{output_path}"
    )
    print()
    print(
        "No existing project source/evaluation files were modified "
        "by this script."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
