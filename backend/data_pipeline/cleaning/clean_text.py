import re

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)        # remove extra newlines
    text = re.sub(r'\s+', ' ', text)         # remove extra spaces
    text = re.sub(r'Page \d+', '', text)     # remove page numbers
    return text.strip()