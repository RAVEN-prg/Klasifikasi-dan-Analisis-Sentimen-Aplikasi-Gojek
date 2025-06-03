import re
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_tokenizer(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def clean_text(text):
    """
    Membersihkan teks untuk preprocessing
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_text(text, tokenizer, maxlen=100):
    """
    Preprocessing teks untuk model LSTM
    """
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
    return padded

def preprocess_text_for_rf(text, vectorizer):
    """
    Preprocessing teks untuk model Random Forest
    """
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    return vectorized_text

def highlight_keywords(text):
    """
    Menyorot kata-kata kunci positif dan negatif dalam teks
    """
    # Daftar kata negatif dan positif (bisa dikembangkan)
    negative_words = [
        "jelek", "lambat", "parah", "buruk", "error", "tidak", "cancel", "gagal",
        "lelet", "lemot", "rusak", "susah", "ribet", "mahal", "boros", "kecewa",
        "mengecewakan", "payah", "kurang", "salah", "telat", "macet", "hang"
    ]
    
    positive_words = [
        "bagus", "cepat", "mantap", "baik", "terbaik", "senang", "puas",
        "mudah", "murah", "hemat", "recommended", "oke", "lancar", "smooth",
        "responsif", "praktis", "efisien", "memuaskan", "excellent", "perfect"
    ]

    # Highlight berdasarkan polaritas
    highlighted_text = text
    
    for word in negative_words:
        pattern = rf"\b({re.escape(word)})\b"
        highlighted_text = re.sub(
            pattern, 
            r"<mark style='background-color: #ffcccc; color: #d32f2f; font-weight: bold;'>\1</mark>", 
            highlighted_text, 
            flags=re.IGNORECASE
        )

    for word in positive_words:
        pattern = rf"\b({re.escape(word)})\b"
        highlighted_text = re.sub(
            pattern, 
            r"<mark style='background-color: #ccffcc; color: #388e3c; font-weight: bold;'>\1</mark>", 
            highlighted_text, 
            flags=re.IGNORECASE
        )

    return highlighted_text

def get_model_info():
    """
    Mengembalikan informasi tentang model yang tersedia
    """
    return {
        "LSTM": {
            "name": "Long Short-Term Memory",
            "type": "Deep Learning",
            "description": "Neural network yang baik untuk memahami konteks dan urutan kata",
            "advantages": ["Memahami konteks", "Baik untuk teks panjang", "Menangani urutan kata"],
            "disadvantages": ["Membutuhkan waktu training lama", "Memerlukan data banyak"]
        },
        "Random Forest": {
            "name": "Random Forest",
            "type": "Machine Learning",
            "description": "Ensemble learning dengan TF-IDF features",
            "advantages": ["Cepat dalam prediksi", "Interpretable", "Tidak memerlukan data banyak"],
            "disadvantages": ["Tidak memahami konteks", "Bergantung pada feature engineering"]
        }
    }

def evaluate_sentiment_strength(confidence_score):
    """
    Mengevaluasi kekuatan sentimen berdasarkan confidence score
    """
    if confidence_score >= 0.8:
        return "Sangat Yakin"
    elif confidence_score >= 0.6:
        return "Yakin"
    elif confidence_score >= 0.4:
        return "Cukup Yakin"
    else:
        return "Kurang Yakin"
