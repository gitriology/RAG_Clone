# scripts/chunk.py
import os
from tqdm import tqdm

def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def chunk_all_texts(input_dir="data/legal/cleaned/"):
    files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    all_chunks = {}
    
    for file in tqdm(files, desc="Chunking texts"):
        with open(os.path.join(input_dir, file), "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text)
        doc_name = file.replace(".txt", "")
        all_chunks[doc_name] = chunks
    
    return all_chunks

if __name__ == "__main__":
    chunks = chunk_all_texts()
    print(f"Sample chunks for first doc: {chunks[list(chunks.keys())[0]][:2]}")