from fastapi import FastAPI
from pydantic import BaseModel
from backend.reranker.scoring.run_rerank import run_pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/api/query")
def query_rag(request: QueryRequest):
    result = run_pipeline(request.query)
    return result