import faiss
import numpy as np

def build_faiss(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index
def search_faiss(query_embedding, index, k=5):
    distances, indices = index.search(query_embedding, k)
    return indices[0]