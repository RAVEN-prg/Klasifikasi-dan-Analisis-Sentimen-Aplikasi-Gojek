import re
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer 

def load_tokenizer(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_vectorizer(path): # Fungsi baru untuk memuat vectorizer RF
    with open(path, "rb") as f:
        return pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_text_lstm(text, tokenizer, maxlen=100):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
    return padded

def preprocess_text_rf(text, vectorizer): # Fungsi baru untuk preprocessing RF
    text = clean_text(text)
    # Pastikan vectorizer.transform() menerima list of strings
    vectorized_text = vectorizer.transform([text])
    return vectorized_text

import re

def highlight_keywords(text):
    # Daftar kata negatif dan positif (bisa dikembangkan)
    negative_words = ["jelek", "lambat", "parah", "buruk", "error", "tidak", "cancel", "gagal", "kecewa"]
    positive_words = ["bagus", "cepat", "mantap", "baik", "terbaik", "senang", "puas", "membantu", "mudah"]

    # Highlight berdasarkan polaritas
    # Iterasi melalui salinan teks asli untuk menghindari masalah dengan modifikasi saat iterasi
    highlighted_text = text
    for word in negative_words:
        # Gunakan re.IGNORECASE untuk mencocokkan tanpa memperhatikan besar kecil huruf
        # Gunakan \b untuk memastikan kata utuh yang dicocokkan
        highlighted_text = re.sub(rf"\b({re.escape(word)})\b", r"<mark style='background-color: #ffcccc;'>\1</mark>", highlighted_text, flags=re.IGNORECASE)

    for word in positive_words:
        highlighted_text = re.sub(rf"\b({re.escape(word)})\b", r"<mark style='background-color: #ccffcc;'>\1</mark>", highlighted_text, flags=re.IGNORECASE)

    return highlighted_text
