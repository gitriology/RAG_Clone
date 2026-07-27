import faiss
from pathlib import Path

from backend.retrieval.utils.load_embeddings import load_embeddings
from backend.retrieval.index.faiss_index import build_faiss


def main():

    print("[FAISS] Loading embeddings...")

    embeddings = load_embeddings()

    print("[FAISS] Building index...")

    index = build_faiss(embeddings)

    output = Path(
        "backend/retrieval/index/faiss.index"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(output)
    )

    print()

    print("[FAISS] Index saved successfully")

    print(output)


if __name__ == "__main__":
    main()