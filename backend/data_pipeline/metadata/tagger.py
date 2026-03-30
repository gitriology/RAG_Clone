def add_metadata(chunks, source="WHO", domain="healthcare"):
    data = []

    for i, chunk in enumerate(chunks):
        data.append({
            "id": i,
            "text": chunk,
            "domain": domain,
            "source": source,
            "type": "medical_guideline"
        })

    return data