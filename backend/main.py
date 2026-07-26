from fastapi import FastAPI
from pydantic import BaseModel
from backend.reranker.scoring.run_rerank import run_pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS (frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request schema
class QueryRequest(BaseModel):
    query: str


# Main API
@app.post("/api/query")
def query_rag(request: QueryRequest):

    results = run_pipeline(request.query)

    if not results:

        return {
            "answer": "Sorry, I couldn't find reliable information for your question.",
            "confidence": 0.0,
            "domain": "unknown",
            "sources": [],
            "meta": {
                "status": "no_answer",
                "pipeline_steps": [
                    "query",
                    "retrieval",
                    "reranking",
                    "validation"
                ]
            }
        }

    top = results[0]

    return {

        "answer": top.get("text", ""),

        "confidence": float(
        top.get("pipeline_confidence", 0.0)
        ),

        "domain": top.get(
            "domain",
            "general"
        ),

        "sources": [
            doc.get("source", "unknown")
            for doc in results
        ],

        "meta": {
            "status": "success",

            "pipeline_steps": [

                "query",

                "retrieval",

                "reranking",

                "selection",

                "generation"

            ]
        }

    }