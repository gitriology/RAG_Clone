import os
import json
import re
from PyPDF2 import PdfReader

BASE_DIR = "domains"

DOMAIN_CONFIG = {
    "administrative": {
        "raw_docs": os.path.join(BASE_DIR, "administrative", "raw_docs"),
        "cleaned_docs": os.path.join(BASE_DIR, "administrative", "cleaned_docs"),
        "chunks": os.path.join(BASE_DIR, "administrative", "chunks"),
        "metadata": os.path.join(BASE_DIR, "administrative", "metadata"),
    },
    "education": {
        "raw_docs": os.path.join(BASE_DIR, "education", "raw_docs"),
        "cleaned_docs": os.path.join(BASE_DIR, "education", "cleaned_docs"),
        "chunks": os.path.join(BASE_DIR, "education", "chunks"),
        "metadata": os.path.join(BASE_DIR, "education", "metadata"),
    }
}


def ensure_folders():
    for domain in DOMAIN_CONFIG.values():
        for folder in domain.values():
            os.makedirs(folder, exist_ok=True)


def clean_text(text):
    text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()


def extract_text_from_pdf(file_path):
    text = ""
    reader = PdfReader(file_path)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()


def chunk_text(text, chunk_size=200, overlap=40):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def split_faq_blocks(text):
    blocks = re.split(r'(?=Q:)', text)
    return [block.strip() for block in blocks if block.strip()]


def get_metadata(domain_name, file_name):
    name = file_name.lower()

    if domain_name == "administrative":
        if "faq" in name:
            return {
                "domain": "administrative",
                "source": "faq",
                "department": "HR",
                "confidentiality": "low"
            }
        elif "employee" in name or "handbook" in name:
            return {
                "domain": "administrative",
                "source": "employee_handbook",
                "department": "HR",
                "confidentiality": "low"
            }
        elif "hr" in name or "manual" in name:
            return {
                "domain": "administrative",
                "source": "hr_manual",
                "department": "HR",
                "confidentiality": "low"
            }
        else:
            return {
                "domain": "administrative",
                "source": "policy",
                "department": "HR",
                "confidentiality": "low"
            }

    if domain_name == "education":
        if "faq" in name:
            return {
                "domain": "education",
                "source": "faq",
                "department": "academic",
                "confidentiality": "low"
            }
        elif "textbook" in name or "education" in name or "ml" in name:
            return {
                "domain": "education",
                "source": "textbook",
                "department": "academic",
                "confidentiality": "low"
            }
        else:
            return {
                "domain": "education",
                "source": "notes",
                "department": "academic",
                "confidentiality": "low"
            }

    return {
        "domain": domain_name,
        "source": "document",
        "department": "general",
        "confidentiality": "low"
    }


def create_chunks(cleaned_text, metadata):
    if metadata["source"] == "faq":
        faq_blocks = split_faq_blocks(cleaned_text)
        final_chunks = []

        for block in faq_blocks:
            if len(block.split()) > 220:
                final_chunks.extend(chunk_text(block, chunk_size=200, overlap=40))
            else:
                final_chunks.append(block)

        return final_chunks

    return chunk_text(cleaned_text, chunk_size=200, overlap=40)


def process_domain(domain_name):
    config = DOMAIN_CONFIG[domain_name]
    raw_folder = config["raw_docs"]
    cleaned_folder = config["cleaned_docs"]
    chunks_folder = config["chunks"]
    metadata_folder = config["metadata"]

    all_chunks = []
    all_metadata = []

    for file_name in sorted(os.listdir(raw_folder)):
        if not (file_name.endswith(".txt") or file_name.endswith(".pdf")):
            continue

        file_path = os.path.join(raw_folder, file_name)

        if file_name.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        elif file_name.endswith(".pdf"):
            raw_text = extract_text_from_pdf(file_path)
        else:
            continue

        if not raw_text.strip():
            print(f"Skipping empty file: {file_name}")
            continue

        cleaned_text = clean_text(raw_text)

        cleaned_file_name = f"{os.path.splitext(file_name)[0]}_cleaned.txt"
        cleaned_file_path = os.path.join(cleaned_folder, cleaned_file_name)
        with open(cleaned_file_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        metadata = get_metadata(domain_name, file_name)
        chunks = create_chunks(cleaned_text, metadata)

        file_chunks = []

        for i, chunk in enumerate(chunks, start=1):
            chunk_data = {
                "chunk_id": f"{os.path.splitext(file_name)[0]}_{i:03d}",
                "chunk_index": i,
                "text": chunk,
                "word_count": len(chunk.split()),
                "source_file": file_name,
                "domain": metadata["domain"],
                "source": metadata["source"],
                "department": metadata["department"],
                "confidentiality": metadata["confidentiality"]
            }
            file_chunks.append(chunk_data)
            all_chunks.append(chunk_data)

        chunk_file_name = f"{os.path.splitext(file_name)[0]}_chunks.json"
        chunk_file_path = os.path.join(chunks_folder, chunk_file_name)

        with open(chunk_file_path, "w", encoding="utf-8") as f:
            json.dump(file_chunks, f, indent=4, ensure_ascii=False)

        metadata_entry = {
            "file_name": file_name,
            "domain": metadata["domain"],
            "source": metadata["source"],
            "department": metadata["department"],
            "confidentiality": metadata["confidentiality"],
            "total_chunks": len(file_chunks)
        }
        all_metadata.append(metadata_entry)

    combined_chunks_path = os.path.join(chunks_folder, f"{domain_name}_all_chunks.json")
    with open(combined_chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=4, ensure_ascii=False)

    metadata_path = os.path.join(metadata_folder, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=4, ensure_ascii=False)

    print(f"{domain_name.capitalize()} domain processed successfully!")
    print(f"Files processed: {len(all_metadata)}")
    print(f"Total chunks created: {len(all_chunks)}\n")


def main():
    ensure_folders()

    for domain_name in DOMAIN_CONFIG:
        process_domain(domain_name)

    print("Phase 1 processing completed for all domains.")


if __name__ == "__main__":
    main()