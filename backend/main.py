from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from backend.reranker.scoring.run_rerank import run_pipeline
from backend.llm_service import GROQ_MODEL, generate_llm_answer, normalize_history, plan_query

import os

PIPELINE_VERSION = "confidence-calibration-v3"

app = FastAPI(title="RAG API", version=PIPELINE_VERSION)

FRONTEND_ORIGIN = os.getenv(
    "FRONTEND_ORIGIN",
    "http://localhost:5173",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from Vercel
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, OPTIONS, etc.
    allow_headers=["*"],  # CRITICAL: Allows ngrok-skip-browser-warning header
)


class ConversationTurn(BaseModel):
    role: str
    text: str


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k_documents: int = Field(default=3, ge=1, le=20)
    max_sentences: int = Field(default=3, ge=1, le=20)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _retrieval_state_from_result(result):
    state = result.get("retrieval_state") or {}
    return {
        "agreement": safe_float(state.get("agreement", 0.0)),
        "complexity": safe_float(state.get("complexity", 0.0)),
        "margin": safe_float(state.get("margin", 0.0)),
        "stability": safe_float(state.get("stability", 0.0)),
        "recommended_top_k": int(state.get("recommended_top_k", 0) or 0),
        "retrieval_confidence": safe_float(
            state.get("retrieval_confidence", result.get("retrieval_confidence", 0.0))
        ),
        "query_type": str(state.get("query_type", "") or ""),
        "decision": str(state.get("decision", "") or ""),
        "decision_reason": str(state.get("decision_reason", "") or ""),
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "pipeline_version": PIPELINE_VERSION}


@app.post("/api/query")
def query_rag(request: QueryRequest):
    history = normalize_history(
        [turn.model_dump() for turn in request.conversation_history]
    )

    # The planner resolves conversational references and identifies genuinely
    # independent topics. The existing RAG pipeline then runs independently for
    # each search query; its internal retrieval algorithms are unchanged.
    plan = plan_query(request.query.strip(), history)
    search_queries = plan.get("search_queries") or [request.query.strip()]
    search_queries = [str(item).strip() for item in search_queries if str(item).strip()][:3]
    if not search_queries:
        search_queries = [request.query.strip()]

    all_results = []
    query_batches = []
    for search_query in search_queries:
        batch = run_pipeline(
            search_query,
            top_k_documents=request.top_k_documents,
            max_sentences=request.max_sentences,
        )
        query_batches.append({"query": search_query, "results": batch or []})
        all_results.extend(batch or [])

    if not all_results:
        return {
            "answer": "I don't know based on the available knowledge base.",
            "confidence": 0.0,
            "domain": "unknown",
            "sources": [],
            "pipeline_version": PIPELINE_VERSION,
            "meta": {
                "status": "no_answer",
                "query": request.query.strip(),
                "standalone_query": plan.get("standalone_query", request.query.strip()),
                "search_queries": search_queries,
                "top_k_documents": request.top_k_documents,
                "max_sentences": request.max_sentences,
                "llm_used": False,
                "llm_model": GROQ_MODEL,
                "conversation_context_used": bool(history),
            },
        }

    # Merge the already-retrieved context from each independent search query.
    # This is what prevents one topic from hiding another topic's documents.
    seen_docs = set()
    merged_context_documents = []
    merged_selected_evidence = []
    seen_evidence = set()
    sources = []
    batch_summaries = []

    for batch in query_batches:
        batch_results = batch["results"]
        batch_sources = []
        batch_confidences = []

        for result in batch_results:
            source = str(result.get("source", "unknown")).strip()
            if source and source not in sources:
                sources.append(source)
            if source and source not in batch_sources:
                batch_sources.append(source)

            # Include every final top-K source from this retrieval batch, not
            # just the single highest-ranked source. This makes multi-topic
            # retrieval visible in the UI as well as available to the LLM.
            for source_document in result.get("source_documents", []) or []:
                source_name = str(source_document.get("source", "")).strip()
                if source_name and source_name not in sources:
                    sources.append(source_name)
                if source_name and source_name not in batch_sources:
                    batch_sources.append(source_name)

            batch_confidences.append(safe_float(result.get("pipeline_confidence", 0.0)))

            for document in result.get("llm_context_documents", []) or []:
                key = (str(document.get("doc_id", "")), str(document.get("source", "")))
                if key in seen_docs:
                    continue
                seen_docs.add(key)
                merged_context_documents.append(document)

            for evidence_item in result.get("selected_evidence", []) or []:
                if isinstance(evidence_item, dict):
                    key = (
                        str(evidence_item.get("source", "")),
                        str(evidence_item.get("text", "")),
                    )
                else:
                    key = ("", str(evidence_item))
                if key in seen_evidence:
                    continue
                seen_evidence.add(key)
                merged_selected_evidence.append(evidence_item)

        batch_summaries.append({
            "query": batch["query"],
            "source_count": len(batch_sources),
            "sources": batch_sources,
            "confidence": max(batch_confidences, default=0.0),
        })

    # Keep the first batch as the primary Evidence Graph/State representation.
    # For multi-topic requests, all retrieval batches remain visible in meta.
    primary = (
        query_batches[0]["results"][0]
        if query_batches[0]["results"]
        else all_results[0]
    )
    rag_answer = str(primary.get("answer", primary.get("text", ""))).strip()

    # The frontend Retrieval State must display MS-ARC retrieval signals, not
    # answer-validation signals. For multi-query requests, use a conservative
    # aggregate: weakest confidence/agreement/margin/stability, highest
    # complexity/top-k requirement. Per-query values remain available in
    # retrieval_batches for diagnostics.
    primary_retrieval_state = _retrieval_state_from_result(primary)
    batch_states = [
        _retrieval_state_from_result(batch_result)
        for batch in query_batches
        for batch_result in batch["results"][:1]
    ]
    if batch_states:
        retrieval_state_summary = {
            "agreement": min(x["agreement"] for x in batch_states),
            "complexity": max(x["complexity"] for x in batch_states),
            "margin": min(x["margin"] for x in batch_states),
            "stability": min(x["stability"] for x in batch_states),
            "recommended_top_k": max(x["recommended_top_k"] for x in batch_states),
            "retrieval_confidence": min(x["retrieval_confidence"] for x in batch_states),
            "query_type": primary_retrieval_state["query_type"],
            "decision": primary_retrieval_state["decision"],
            "decision_reason": primary_retrieval_state["decision_reason"],
            "aggregation": "conservative_multi_query",
            "per_query": batch_states,
        }
    else:
        retrieval_state_summary = primary_retrieval_state

    llm_answer = generate_llm_answer(
        request.query.strip(),
        merged_selected_evidence,
        context_documents=merged_context_documents[:6],
        conversation_history=history,
        fallback_answer=rag_answer,
    )

    confidence_values = [
        safe_float(result.get("pipeline_confidence", 0.0))
        for result in all_results
    ]
    # Conservative aggregate: a multi-topic answer should not claim higher
    # confidence than its weakest retrieved topic.
    confidence = min(confidence_values) if confidence_values else 0.0

    graph = primary.get("evidence_graph", {}) or {}
    evidence_state = primary.get("evidence_state", {}) or {}

    return {
        "answer": llm_answer,
        "confidence": confidence,
        "domain": primary.get("domain", "general"),
        "sources": sources,
        "pipeline_version": primary.get("pipeline_version", PIPELINE_VERSION),
        "evidence_graph": graph,
        "evidence_state": evidence_state,
        "selected_evidence": merged_selected_evidence,
        "retrieved_documents": merged_context_documents[:6],
        # Expose both confidence layers at the top level as well as in meta.
        # This makes the response contract explicit and keeps the frontend
        # independent of whether a message is newly generated or restored.
        "pipeline_confidence": confidence,
        "retrieval_confidence": retrieval_state_summary["retrieval_confidence"],
        "retrieval_state": retrieval_state_summary,
        "meta": {
            "status": "low_confidence" if confidence < 0.40 else "success",
            "query": request.query.strip(),
            "standalone_query": plan.get("standalone_query", request.query.strip()),
            "search_queries": search_queries,
            "top_k_documents": request.top_k_documents,
            "max_sentences": request.max_sentences,
            "confidence_calibration": primary.get("confidence_calibration", {}),
            "retrieval_confidence": retrieval_state_summary["retrieval_confidence"],
            "answer_confidence": safe_float(primary.get("answer_confidence", 0.0)),
            # Correct MS-ARC retrieval agreement. This must not be mapped from
            # answer_confidence/answer_agreement.
            "agreement": retrieval_state_summary["agreement"],
            "complexity": retrieval_state_summary["complexity"],
            "margin": retrieval_state_summary["margin"],
            "stability": retrieval_state_summary["stability"],
            "recommended_top_k": retrieval_state_summary["recommended_top_k"],
            "retrieval_state": retrieval_state_summary,
            "answer_valid": bool(primary.get("answer_valid", False)),
            "pipeline_confidence": confidence,
            "llm_used": llm_answer != rag_answer,
            "llm_model": GROQ_MODEL,
            "conversation_context_used": bool(history),
            "multi_query": len(search_queries) > 1,
            "retrieval_batches": batch_summaries,
            "retrieved_document_count": len(merged_context_documents),
            "optimization_21": {
                "evidence_graph_integrated": bool(graph),
                "evidence_state_integrated": bool(evidence_state),
                "graph_nodes": int(graph.get("node_count", 0)),
                "graph_edges": int(graph.get("edge_count", 0)),
                "evidence_features": int(evidence_state.get("feature_count", 0)),
                "evidence_score": safe_float(evidence_state.get("evidence_score", 0.0)),
            },
        },
    }

