import os
from ingestion.load_pdf import load_pdf
from cleaning.clean_text import clean_text
from chunking.chunker import chunk_text
from metadata.tagger import add_metadata
from utils.save_json import save_json

INPUT_DIR = "data/raw/healthcare"
OUTPUT_FILE = "data/processed/healthcare_chunks.json"

all_data = []

for file in os.listdir(INPUT_DIR):
    if file.endswith(".pdf"):
        file_path = os.path.join(INPUT_DIR, file)

        print(f"Processing: {file}")

        # Step 1: Load
        text = load_pdf(file_path)

        # Step 2: Clean
        cleaned = clean_text(text)

        # Step 3: Chunk
        chunks = chunk_text(cleaned)

        # Step 4: Metadata
        data = add_metadata(chunks, source=file)

        all_data.extend(data)

# Step 5: Save
save_json(all_data, OUTPUT_FILE)

print("Phase 1 Completed Successfully!")