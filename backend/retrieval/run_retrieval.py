from backend.ms_arc.run_msarc import run_msarc


def retrieve(query):

    state = run_msarc(query)

    documents = []

    for doc in state.reranked_results if state.reranked_results else state.selected_documents:

        documents.append({

            "text": doc.text,

            "domain": doc.metadata.get("domain", "general"),

            "source": doc.metadata.get("source", "unknown"),

            "rerank_score": doc.rerank_score,

            "dense_score": doc.dense_score,

            "sparse_score": doc.sparse_score,

            "hybrid_score": doc.metadata.get("hybrid_score", 0.0)

        })

    return {

        "documents": documents,

        "retrieval_confidence": state.retrieval_confidence,

        "evidence": {

            "agreement": state.signals.agreement.score,

            "margin": state.signals.margin.normalized_margin,

            "stability": state.signals.stability.score,

            "decision": state.signals.decision.decision,

            "reason": state.signals.decision.reason

        }

    }