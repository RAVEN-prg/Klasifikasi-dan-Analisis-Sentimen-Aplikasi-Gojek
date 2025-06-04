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

def highlight_keywords_rf(text, top_features):
    """
    Highlight keywords based on Random Forest's important features
    top_features should be a dict with 'positive' and 'negative' keys
    containing lists of important words
    """
    # Highlight positive words
    for word in top_features['positive']:
        text = re.sub(rf"\b({re.escape(word)})\b", 
                     r"<mark style='background-color: #ccffcc;'>\1</mark>", 
                     text, flags=re.IGNORECASE)
    
    # Highlight negative words
    for word in top_features['negative']:
        text = re.sub(rf"\b({re.escape(word)})\b", 
                     r"<mark style='background-color: #ffcccc;'>\1</mark>", 
                     text, flags=re.IGNORECASE)
    
    return text
