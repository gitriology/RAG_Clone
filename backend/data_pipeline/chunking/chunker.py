def chunk_text(text, chunk_size=400, overlap=80):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        
        # skip very small chunks
        if len(chunk.split()) > 50:
            chunks.append(chunk)

    return chunks