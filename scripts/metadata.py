# scripts/metadata.py

def add_metadata(all_chunks, metadata_dict):
    all_data = []
    for doc_name, chunks in all_chunks.items():
        meta = metadata_dict.get(doc_name, {})
        for i, chunk in enumerate(chunks):
            item = {
                "chunk_id": f"{doc_name}_{i}",
                "text": chunk,
                "domain": "legal",
                "subdomain": meta.get("subdomain", ""),
                "doc_type": meta.get("doc_type", "")
            }
            all_data.append(item)
    return all_data