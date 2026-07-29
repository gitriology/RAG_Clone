import re

from backend.retrieval.run_retrieval import retrieve
from backend.reranker.model.reranker import rerank
from backend.reranker.utils.context_selector import select_top_k
from backend.generation.validation.context_validator import validate_context
from backend.generation.guards.answer_validator import validate_answer
from backend.reranker.scoring.confidence.confidence_score import compute_confidence
from backend.optimization.confidence_margin import confidence_margin


def clean_text(text):
    # Remove URLs (normal + broken ones)
    text = re.sub(r'http\S+|www\S+', '', text)

    # Fix broken spacing like "thehindu. c om"
    text = re.sub(r'\s+\.\s+', '.', text)

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def build_answer(docs):
    answer_parts = []

    for doc in docs:
        text = clean_text(doc["text"])

        # Split using safer boundary (not naive ".")
        chunks = re.split(r'(?<=[.!?])\s+', text)

        # Filter out garbage chunks
        chunks = [
            c.strip()
            for c in chunks
            if c.strip() and len(c.split()) > 3  # ignore tiny/noisy fragments
        ]

        if chunks:
            # Take first 2–3 meaningful chunks
            selected = chunks[:2]

            # IMPORTANT: do NOT force "." — keep original text
            answer_parts.extend(selected)

    return " ".join(answer_parts)


def run_pipeline(query):
    # Step 1: Retrieval
    retrieved_docs = retrieve(query)

    # Step 2: Reranking
    ranked_docs = rerank(query, retrieved_docs)
    margin = confidence_margin(ranked_docs)
    print(f"\nCross-Encoder Confidence Margin : {margin:.3f}")

    # Step 3: Context validation
    validated_docs = validate_context(query, ranked_docs)

    # Step 4: Select top docs
    top_docs = select_top_k(validated_docs, k=3)

    # Step 5: Build answer
    answer = build_answer(top_docs)

    # Step 6: Answer validation
    validation = validate_answer(answer, top_docs)

    # Step 7: Confidence scoring
    confidence = compute_confidence(top_docs, validation)

    # ✅ NEW: format output for API
    formatted_results = []

    for doc in top_docs:
        formatted_results.append({
            "text": answer,  # final generated answer
            "rerank_score": float(doc.get("rerank_score", 0.0)),
            "domain": doc.get("domain", "general"),
            "source": doc.get("source", "unknown")
        })

    return formatted_results

if __name__ == "__main__":
    query = "Where did Chandrayaan-3 land on the Moon and which organization developed the mission?"

    result = run_pipeline(query)

    print("\nRESULT:\n")

    for i, doc in enumerate(result):
        print(f"\n--- Document {i+1} ---")
        print("Answer:", doc["text"])
        print("Score:", doc["rerank_score"])
        print("Domain:", doc["domain"])
        print("Source:", doc["source"])