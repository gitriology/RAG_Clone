import os
import json
from src.extraction import extract_text_from_pdf
from src.cleaning import clean_text
from src.chunking import chunk_text
from src.metadata import add_metadata

INPUT_FOLDER = "data/raw"
OUTPUT_FILE = "data/processed/output.json"

all_data = []

for file in os.listdir(INPUT_FOLDER):
    if file.endswith(".pdf"):
        file_path = os.path.join(INPUT_FOLDER, file)

        print(f"Processing {file}...")

        text = extract_text_from_pdf(file_path)
        clean = clean_text(text)
        chunks = chunk_text(clean)
        metadata = add_metadata(chunks, file)

        all_data.extend(metadata)

# Save output
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=4)

print("✅ Done! Data saved to output.json")