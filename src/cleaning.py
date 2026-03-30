import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    text = re.sub(r'\[\d+\]', '', text)  # remove references like [1], [2]
    return text.strip()