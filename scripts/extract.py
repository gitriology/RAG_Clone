# scripts/extract.py
import fitz  # PyMuPDF
import os
from tqdm import tqdm

def extract_pdf(input_path, output_path):
    doc = fitz.open(input_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

def extract_all_pdfs(pdf_dir="data/legal/raw/", txt_dir="data/legal/extracted/"):
    os.makedirs(txt_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    
    for pdf_file in tqdm(pdf_files, desc="Extracting PDFs"):
        input_path = os.path.join(pdf_dir, pdf_file)
        output_path = os.path.join(txt_dir, pdf_file.replace(".pdf", ".txt"))
        extract_pdf(input_path, output_path)

if __name__ == "__main__":
    extract_all_pdfs()