import json
import os

def normalize_chunk(chunk, idx):
    normalized = {
        "id": idx,
        "text": "",
        "domain": "unknown",
        "source": "unknown",
        "type": "general"
    }

    # TEXT
    if "text" in chunk:
        normalized["text"] = chunk["text"]

    # SPACE DOMAIN (has metadata)
    if "metadata" in chunk:
        metadata = chunk["metadata"]
        normalized["domain"] = metadata.get("domain", "space")
        normalized["source"] = metadata.get("source", "unknown")
        normalized["type"] = metadata.get("department", "research")

    # LEGAL DOMAIN
    elif "chunk_id" in chunk:
        normalized["domain"] = chunk.get("domain", "legal")
        normalized["source"] = chunk.get("chunk_id", "legal_doc")
        normalized["type"] = chunk.get("doc_type", "legal")

    # HEALTHCARE DOMAIN
    else:
        normalized["domain"] = chunk.get("domain", "healthcare")
        normalized["source"] = chunk.get("source", "unknown")
        normalized["type"] = chunk.get("type", "medical")

    return normalized


def normalize_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized_data = []

    for idx, chunk in enumerate(data):
        norm = normalize_chunk(chunk, idx)

        if norm["text"].strip():
            normalized_data.append(norm)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Normalized: {input_path}")


def combine_all(files, output_path):
    combined = []

    for file in files:
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                combined.extend(json.load(f))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print("✅ Combined dataset created!")


if __name__ == "__main__":
    base_path = "data/processed/"

    files = {
        "healthcare": base_path + "healthcare_chunks.json",
        "legal": base_path + "legal_chunks.json",
        "space": base_path + "space_chunks.json"
    }

    clean_files = []

    for domain, path in files.items():
        output_path = base_path + f"{domain}_clean.json"
        normalize_file(path, output_path)
        clean_files.append(output_path)

    combine_all(clean_files, base_path + "all_domains.json")

    print("🎉 All domains normalized and combined!")