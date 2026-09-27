from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from backend.reranker.scoring.run_rerank import run_pipeline

import os

PIPELINE_VERSION = "confidence-calibration-v3"

app = FastAPI(title="RAG API", version=PIPELINE_VERSION)

FRONTEND_ORIGIN = os.getenv(
    "FRONTEND_ORIGIN",
    "http://localhost:5173",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k_documents: int = Field(default=3, ge=1, le=20)
    max_sentences: int = Field(default=3, ge=1, le=20)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@app.get("/api/health")
def health():
    return {"status": "ok", "pipeline_version": PIPELINE_VERSION}


@app.post("/api/query")
def query_rag(request: QueryRequest):
    results = run_pipeline(
        request.query,
        top_k_documents=request.top_k_documents,
        max_sentences=request.max_sentences,
    )

    if not results:
        return {
            "answer": "Sorry, I couldn't find reliable information for your question.",
            "confidence": 0.0,
            "domain": "unknown",
            "sources": [],
            "pipeline_version": PIPELINE_VERSION,
            "meta": {
                "status": "no_answer",
                "query": request.query.strip(),
                "top_k_documents": request.top_k_documents,
                "max_sentences": request.max_sentences,
            },
        }

    top = results[0]
    sources = []
    for doc in results:
        source = str(doc.get("source", "unknown")).strip()
        if source and source not in sources:
            sources.append(source)

    graph = top.get("evidence_graph", {}) or {}
    evidence_state = top.get("evidence_state", {}) or {}
    confidence = safe_float(top.get("pipeline_confidence", 0.0))

    return {
        "answer": str(top.get("answer", top.get("text", ""))).strip(),
        "confidence": confidence,
        "domain": top.get("domain", "general"),
        "sources": sources,
        "pipeline_version": top.get("pipeline_version", PIPELINE_VERSION),
        "evidence_graph": graph,
        "evidence_state": evidence_state,
        "selected_evidence": top.get("selected_evidence", []),
        "meta": {
            "status": "low_confidence" if confidence < 0.40 else "success",
            "query": top.get("query", request.query.strip()),
            "top_k_documents": request.top_k_documents,
            "max_sentences": request.max_sentences,
            "confidence_calibration": top.get("confidence_calibration", {}),
            "retrieval_confidence": safe_float(top.get("retrieval_confidence", 0.0)),
            "answer_confidence": safe_float(top.get("answer_confidence", 0.0)),
            "answer_agreement": safe_float(top.get("answer_confidence", 0.0)),
            "answer_valid": bool(top.get("answer_valid", False)),
            "pipeline_confidence": confidence,
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
