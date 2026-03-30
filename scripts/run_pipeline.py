# scripts/run_pipeline.py
from extract import extract_all_pdfs
from clean import clean_all_texts
from chunk import chunk_all_texts
from metadata import add_metadata
import json

# 1️⃣ Extract
extract_all_pdfs()

# 2️⃣ Clean
clean_all_texts()

# 3️⃣ Chunk
all_chunks = chunk_all_texts()

# 4️⃣ Metadata
metadata_dict = {
    "companies_act": {"subdomain": "corporate_law", "doc_type": "act"},
    "gfr": {"subdomain": "public_finance", "doc_type": "rules"},
    "rbi_guidelines": {"subdomain": "banking", "doc_type": "guideline"}
}
all_data = add_metadata(all_chunks, metadata_dict)

# 5️⃣ Save JSON
with open("data/legal/processed/legal_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2)

print("✅ Phase-1 pipeline complete!")