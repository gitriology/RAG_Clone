from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from backend.reranker.scoring.run_rerank import (
    run_pipeline,
)


# ==========================================================
# APPLICATION
# ==========================================================

app = FastAPI()


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# REQUEST SCHEMA
# ==========================================================

class QueryRequest(BaseModel):

    query: str


# ==========================================================
# SAFE FLOAT
# ==========================================================

def safe_float(
    value,
    default=0.0,
):
    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


# ==========================================================
# MAIN API
# ==========================================================

@app.post("/api/query")
def query_rag(
    request: QueryRequest,
):

    results = run_pipeline(
        request.query
    )

    # ======================================================
    # NO RESULT
    # ======================================================

    if not results:

        return {
            "answer":
                "Sorry, I couldn't find reliable "
                "information for your question.",

            "confidence":
                0.0,

            "domain":
                "unknown",

            "sources":
                [],

            "meta": {

                "status":
                    "no_answer",

                "pipeline_steps": [

                    "query",

                    "retrieval",

                    "reranking",

                    "evidence_graph",

                    "evidence_state",

                    "validation",

                ],
            },
        }

    # ======================================================
    # TOP RESULT
    # ======================================================

    top = results[0]

    # ======================================================
    # SOURCE DOCUMENTS
    # ======================================================

    sources = []

    for doc in results:

        source = doc.get(
            "source",
            "unknown",
        )

        if source not in sources:

            sources.append(
                source
            )

    # ======================================================
    # EVIDENCE GRAPH
    # ======================================================

    graph = top.get(
        "evidence_graph",
        {},
    )

    # ======================================================
    # EVIDENCE STATE
    # ======================================================

    evidence_state = top.get(
        "evidence_state",
        {},
    )

    # ======================================================
    # RESPONSE
    # ======================================================

    return {

        "answer":
            top.get(
                "text",
                "",
            ),

        "confidence":
            safe_float(
                top.get(
                    "pipeline_confidence",
                    0.0,
                )
            ),

        "domain":
            top.get(
                "domain",
                "general",
            ),

        "sources":
            sources,

        # ==================================================
        # Research / Diagnostics
        # ==================================================

        "evidence_graph":
            graph,

        "evidence_state":
            evidence_state,

        # ==================================================
        # Metadata
        # ==================================================

        "meta": {

            "status":
                "success",

            "pipeline_steps": [

                "query",

                "retrieval",

                "reranking",

                "evidence_graph",

                "evidence_state",

                "selection",

                "generation",

                "validation",

            ],

            "optimization_21":
                {

                    "evidence_graph_integrated":
                        bool(
                            graph
                        ),

                    "evidence_state_integrated":
                        bool(
                            evidence_state
                        ),

                    "graph_nodes":
                        int(
                            graph.get(
                                "node_count",
                                0,
                            )
                        ),

                    "graph_edges":
                        int(
                            graph.get(
                                "edge_count",
                                0,
                            )
                        ),

                    "evidence_features":
                        int(
                            evidence_state.get(
                                "feature_count",
                                0,
                            )
                        ),

                    "evidence_score":
                        safe_float(
                            evidence_state.get(
                                "evidence_score",
                                0.0,
                            )
                        ),
                },
        },
    }