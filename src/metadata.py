def add_metadata(chunks, source):
    data = []

    for i, chunk in enumerate(chunks):
        data.append({
            "text": chunk,
            "metadata": {
                "domain": "space_missions",
                "source": source,
                "department": "research",
                "confidentiality": "public"
            }
        })

    return data