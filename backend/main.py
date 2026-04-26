# from fastapi import FastAPI
# from pydantic import BaseModel
# from backend.reranker.scoring.run_rerank import run_pipeline
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class QueryRequest(BaseModel):
#     query: str

# @app.post("/api/query")
# def query_rag(request: QueryRequest):
#     result = run_pipeline(request.query)
#     return result

from fastapi import FastAPI
from pydantic import BaseModel
from backend.reranker.scoring.run_rerank import run_pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ Enable CORS (frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Request schema
class QueryRequest(BaseModel):
    query: str


# ✅ Main API
@app.post("/api/query")
def query_rag(request: QueryRequest):
    results = run_pipeline(request.query)

    # ⚠️ Safety fallback
    if not results:
        return {
            "answer": "No relevant information found.",
            "confidence": 0.0,
            "domain": "unknown",
            "sources": [],
            "meta": {"pipeline_steps": []}
        }

    top = results[0]

    return {
        "answer": top.get("text", ""),
        "confidence": float(top.get("rerank_score", 0.0)),
        "domain": top.get("domain", "general"),

        "sources": [
            doc.get("source", "unknown") for doc in results
        ],

        "meta": {
            "pipeline_steps": [
                "query",
                "retrieval",
                "reranking",
                "selection",
                "generation"
            ]
        }
    }