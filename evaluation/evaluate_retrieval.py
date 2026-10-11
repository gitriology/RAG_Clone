#!/usr/bin/env python3
"""Cold/warm retrieval benchmark and optional answer-quality audit.

Run from the project root:
  python evaluation/evaluate_retrieval.py --output evaluation/results.csv --repeats 4
  python evaluation/evaluate_retrieval.py --output evaluation/results.csv --repeats 4 --answer-audit

Timing semantics:
- first_run_seconds is the first request after clearing retrieval caches for that
  query. It may still benefit from already-loaded process/model state.
- warm_* values are subsequent runs with the same query embedding retained but
  result/anchor caches cleared, so retrieval stages are measured rather than a
  trivial full-result cache hit.
- true process cold-start requires a fresh process; this script reports the
  first query separately and records process-start/model-init effects there.
"""
from __future__ import annotations
import argparse, csv, json, statistics, sys, time
from pathlib import Path
from typing import Any

EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATION_DIR.parent
DEFAULT_QUERIES = EVALUATION_DIR / "queries.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def normalize_source(value: Any) -> str:
    value = str(value or "").strip().replace("\\", "/")
    return value.rsplit("/", 1)[-1].casefold()


def recall_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    expected = {normalize_source(x) for x in relevant if str(x).strip()}
    if not expected: return 0.0
    found = {normalize_source(x) for x in ranked[:k]}
    return len(found & expected) / len(expected)


def reciprocal_rank(ranked: list[str], relevant: list[str]) -> float:
    expected = {normalize_source(x) for x in relevant if str(x).strip()}
    for rank, source in enumerate(ranked, 1):
        if normalize_source(source) in expected: return 1.0 / rank
    return 0.0


def topic_coverage(ranked: list[str], topics: list[dict[str, Any]], k: int = 5):
    if not topics: return "N/A"
    applicable = [t for t in topics if t.get("relevant_documents")]
    if not applicable: return "N/A"
    top = {normalize_source(x) for x in ranked[:k]}
    return round(sum(bool(top & {normalize_source(x) for x in t["relevant_documents"]}) for t in applicable) / len(applicable), 4)


def load_queries(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f: data = json.load(f)
    if not isinstance(data, list) or not data: raise ValueError(f"Expected non-empty JSON list: {path}")
    for i, item in enumerate(data, 1):
        if not isinstance(item, dict) or not str(item.get("query", "")).strip(): raise ValueError(f"Query #{i} missing query")
        if not isinstance(item.get("relevant_documents"), list) or not item["relevant_documents"]: raise ValueError(f"Query #{i} missing relevant_documents")
    return data


def source_names(result: dict[str, Any]) -> list[str]:
    state = result.get("retrieval_state")
    docs = getattr(state, "reranked_results", None) if state is not None else None
    if not docs and state is not None: docs = getattr(state, "selected_documents", None)
    if not docs: docs = result.get("documents") or []
    ranked, seen = [], set()
    for doc in docs:
        if isinstance(doc, dict): source = doc.get("source") or doc.get("metadata", {}).get("source")
        else: source = getattr(doc, "source", None) or getattr(doc, "metadata", {}).get("source")
        if not source: continue
        source = str(source).strip(); key = normalize_source(source)
        if key and key != "unknown" and key not in seen:
            ranked.append(source); seen.add(key)
    return ranked


def get_documents(result: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    state = result.get("retrieval_state")
    docs = getattr(state, "reranked_results", None) if state is not None else None
    if not docs and state is not None: docs = getattr(state, "selected_documents", None)
    out=[]
    for doc in (docs or [])[:limit]:
        if isinstance(doc, dict):
            out.append({"source": doc.get("source") or doc.get("metadata", {}).get("source", "unknown"), "text": doc.get("text", ""), "rerank_score": doc.get("rerank_score")})
        else:
            out.append({"source": getattr(doc, "metadata", {}).get("source", "unknown"), "text": getattr(doc, "text", ""), "rerank_score": getattr(doc, "rerank_score", None)})
    return out


def evaluate(queries: list[dict[str, Any]], repeats: int, answer_audit: bool, top_k: int):
    from backend.retrieval.run_retrieval import retrieve
    from backend.retrieval.hybrid.hybrid_search import clear_query_cache, clear_result_cache, query_cache_info
    rows=[]; quality=[]; baseline_rankings={}; all_run_diagnostics={}
    for i, item in enumerate(queries, 1):
        query=str(item["query"]).strip(); relevant=[str(x).strip() for x in item["relevant_documents"] if str(x).strip()]
        print(f"\n[{i}/{len(queries)}] {query}", flush=True)
        durations=[]; results=[]; per_run=[]
        for run_index in range(repeats):
            if run_index == 0:
                clear_query_cache()
            else:
                # Retain the query embedding but force the retrieval stages to
                # execute again, avoiding a full result-cache hit masquerading as speed.
                clear_result_cache()
            start=time.perf_counter(); result=retrieve(query); elapsed=time.perf_counter()-start
            durations.append(elapsed); results.append(result)
            state=result.get("retrieval_state")
            debug=getattr(state,"debug",{}) if state is not None else {}
            per_run.append({"run_number":run_index+1,"run_type":"first_run" if run_index==0 else "warm_embedding_cache","total_seconds":round(elapsed,6),
                "stage_timings_seconds":debug.get("stage_timings_seconds",{}),
                "adaptive_iterations_count":debug.get("adaptive_iterations_count"),
                "candidate_count":debug.get("candidate_count",debug.get("adaptive_documents_considered")),
                "hybrid_candidate_count":debug.get("hybrid_candidate_count"),
                "embedding_cache_status":debug.get("embedding_cache_status"),
                "embedding_cache_diagnostics":debug.get("embedding_cache_diagnostics",query_cache_info())})
        ranked=source_names(results[-1]); first_ranked=source_names(results[0])
        baseline_rankings[str(i)] = {"query":query,"first_run_ranked_sources":first_ranked,"warm_run_ranked_sources":ranked,"ranking_identical":first_ranked==ranked}
        all_run_diagnostics[str(i)] = {"query": query, "runs": per_run}
        warm=durations[1:]
        stage_names = ("query_embedding_seconds", "dense_search_seconds", "bm25_search_seconds", "candidate_fusion_seconds", "reranking_seconds", "hybrid_search_total_seconds", "retrieval_seconds", "retrieval_signals_seconds", "incremental_evidence_seconds", "controller_graph_seconds", "controller_evidence_state_seconds", "ms_arc_total_seconds")
        first_stages = per_run[0].get("stage_timings_seconds", {})
        warm_stage_maps = [r.get("stage_timings_seconds", {}) for r in per_run[1:]]
        stage_csv = {}
        for stage_name in stage_names:
            first_value = first_stages.get(stage_name)
            warm_values = [m[stage_name] for m in warm_stage_maps if isinstance(m.get(stage_name), (int, float))]
            stage_csv[f"first_{stage_name}"] = round(first_value, 6) if isinstance(first_value, (int, float)) else "N/A"
            stage_csv[f"warm_mean_{stage_name}"] = round(statistics.mean(warm_values), 6) if warm_values else "N/A"
        expected={normalize_source(x) for x in relevant}
        rows.append({"query_id":i,"query":query,"expected_documents":" | ".join(relevant),"ranked_sources":" | ".join(ranked),
            "top1":" | ".join(ranked[:1]),"top3":" | ".join(ranked[:3]),"top5":" | ".join(ranked[:5]),
            "recall_at_1":round(recall_at_k(ranked,relevant,1),4),"recall_at_3":round(recall_at_k(ranked,relevant,3),4),
            "recall_at_5":round(recall_at_k(ranked,relevant,5),4),"mrr":round(reciprocal_rank(ranked,relevant),4),
            "topic_coverage_at_5":topic_coverage(ranked,item.get("topics") or [],5),
            "retrieved_relevant_count_at_5":len({normalize_source(x) for x in ranked[:5]} & expected),"expected_relevant_count":len(expected),
            "first_run_seconds":round(durations[0],3),"warm_mean_seconds":round(statistics.mean(warm),3) if warm else "N/A",
            "warm_min_seconds":round(min(warm),3) if warm else "N/A","warm_max_seconds":round(max(warm),3) if warm else "N/A",
            "ranking_stable_across_runs":first_ranked==ranked,"adaptive_iterations":per_run[-1]["adaptive_iterations_count"],"candidate_count":per_run[-1]["candidate_count"],
            "query_embedding_cache_hits":per_run[-1]["embedding_cache_diagnostics"].get("embedding_cache_hits",0),
            "query_embedding_cache_misses":per_run[-1]["embedding_cache_diagnostics"].get("embedding_cache_misses",0), **stage_csv})
        print("  First run: %.3fs | warm mean: %s | ranking stable: %s" % (durations[0], f"{statistics.mean(warm):.3f}s" if warm else "N/A", first_ranked==ranked), flush=True)
        if answer_audit:
            # Generate answers only after timing retrieval; generation time is
            # separately reported and does not contaminate retrieval latency.
            from backend.generation.answer_generator import generate_answer
            docs=get_documents(results[-1], limit=top_k)
            answer_start=time.perf_counter(); generated=generate_answer(query, docs, max_sentences=3); answer_seconds=time.perf_counter()-answer_start
            selected=generated.get("selected_sentences",[])
            answer_sources=sorted({str(x.get("source", "unknown")) for x in selected if x.get("source")})
            expected_normalized={normalize_source(x) for x in relevant}
            for passage in docs:
                passage["expected_source_match"] = normalize_source(passage.get("source")) in expected_normalized
            quality.append({"query_id":i,"query":query,"expected_documents":relevant,"ranked_sources":ranked,"top_passages":docs,
                "generated_answer":generated.get("answer",""),"selected_evidence":selected,"generation_seconds":round(answer_seconds,3),
                "best_similarity":generated.get("best_similarity"),"best_evidence_score":generated.get("best_evidence_score"),
                "selected_evidence_count":generated.get("selected_count",len(selected)),
                "review_status":"REQUIRES_HUMAN_REVIEW","review_note":"Check factual correctness, completeness, and whether every answer claim is supported by selected evidence."})
    return rows, all_run_diagnostics, quality, baseline_rankings


def main() -> int:
    parser=argparse.ArgumentParser(description="Cold/warm retrieval benchmark with optional answer-quality audit")
    parser.add_argument("--queries",type=Path,default=DEFAULT_QUERIES); parser.add_argument("--output",type=Path,default=EVALUATION_DIR/"results.csv")
    parser.add_argument("--repeats",type=int,default=4,help="Runs per query; first run separated from warm runs")
    parser.add_argument("--top-k",type=int,default=5); parser.add_argument("--answer-audit",action="store_true",help="Generate answers separately and save evidence for human quality review")
    args=parser.parse_args()
    if args.repeats<2: parser.error("--repeats must be at least 2 to compare first and warm runs")
    if args.top_k<1: parser.error("--top-k must be positive")
    queries=load_queries(args.queries); rows, run_diagnostics, quality, rankings=evaluate(queries,args.repeats,args.answer_audit,args.top_k)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary_path=args.output.with_name(args.output.stem+"_diagnostics.json")
    summary={"benchmark_semantics":"first run after clearing retrieval caches; warm runs retain query embeddings but clear full result/anchor caches",
        "repeats_per_query":args.repeats,"queries":len(rows),"mean_recall_at_1":statistics.mean(r["recall_at_1"] for r in rows),
        "mean_recall_at_3":statistics.mean(r["recall_at_3"] for r in rows),"mean_recall_at_5":statistics.mean(r["recall_at_5"] for r in rows),
        "mean_mrr":statistics.mean(r["mrr"] for r in rows),"mean_first_run_seconds":statistics.mean(r["first_run_seconds"] for r in rows),
        "mean_warm_seconds":statistics.mean(r["warm_mean_seconds"] for r in rows if isinstance(r["warm_mean_seconds"],(int,float))),
        "all_rankings_stable":all(r["ranking_stable_across_runs"] for r in rows),"rankings":rankings,"per_query_runs":run_diagnostics}
    summary_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    if args.answer_audit:
        quality_path=args.output.with_name(args.output.stem+"_answer_quality_audit.json")
        quality_path.write_text(json.dumps(quality,indent=2,ensure_ascii=False),encoding="utf-8")
        print(f"Answer-quality review artifact: {quality_path.resolve()}")
    print(f"\nMean Recall@1: {summary['mean_recall_at_1']:.4f}; Recall@3: {summary['mean_recall_at_3']:.4f}; Recall@5: {summary['mean_recall_at_5']:.4f}; MRR: {summary['mean_mrr']:.4f}")
    print(f"Mean first-run latency: {summary['mean_first_run_seconds']:.3f}s; mean warm latency: {summary['mean_warm_seconds']:.3f}s")
    print(f"All rankings stable across first/warm runs: {summary['all_rankings_stable']}")
    print(f"Saved CSV: {args.output.resolve()}\nSaved diagnostics: {summary_path.resolve()}")
    return 0
if __name__=="__main__": raise SystemExit(main())
