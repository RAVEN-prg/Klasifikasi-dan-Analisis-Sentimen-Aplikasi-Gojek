import re
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_tokenizer(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_text(text, tokenizer, maxlen=100):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
    return padded

import re

def highlight_keywords(text):
    # Daftar kata negatif dan positif (bisa dikembangkan)
    negative_words = ["jelek", "lambat", "parah", "buruk", "error", "tidak", "cancel", "gagal"]
    positive_words = ["bagus", "cepat", "mantap", "baik", "terbaik", "senang", "puas"]

    # Highlight berdasarkan polaritas
    for word in negative_words:
        text = re.sub(rf"\b({word})\b", r"<mark style='background-color: #ffcccc;'>\1</mark>", text, flags=re.IGNORECASE)

    for word in positive_words:
        text = re.sub(rf"\b({word})\b", r"<mark style='background-color: #ccffcc;'>\1</mark>", text, flags=re.IGNORECASE)

    return text
