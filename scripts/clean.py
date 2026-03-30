# scripts/clean.py
import re
import os
from tqdm import tqdm

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)   # remove extra newlines
    text = re.sub(r'\s+', ' ', text)    # remove extra spaces
    return text.strip()

def clean_all_texts(input_dir="data/legal/extracted/", output_dir="data/legal/cleaned/"):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    
    for file in tqdm(files, desc="Cleaning texts"):
        with open(os.path.join(input_dir, file), "r", encoding="utf-8") as f:
            text = f.read()
        cleaned = clean_text(text)
        with open(os.path.join(output_dir, file), "w", encoding="utf-8") as f:
            f.write(cleaned)

if __name__ == "__main__":
    clean_all_texts()