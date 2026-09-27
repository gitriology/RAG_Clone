FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/opt/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt

RUN pip install --no-cache-dir -r ./backend/requirements.txt

RUN python -m spacy download en_core_web_sm

COPY backend ./backend
COPY data/processed/all_domains.json ./data/processed/all_domains.json

RUN python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder

print("Downloading BGE embedding model...")
SentenceTransformer("BAAI/bge-base-en-v1.5")

print("Downloading MiniLM validation model...")
SentenceTransformer("all-MiniLM-L6-v2")

print("Downloading CrossEncoder model...")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("All models downloaded successfully.")
PY

RUN python -m backend.retrieval.utils.save_embeddings

RUN python -m backend.retrieval.index.build_index

RUN python -c "import backend.ms_arc.retrieval.retrieve; print('Production retrieval initialization OK')"

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]