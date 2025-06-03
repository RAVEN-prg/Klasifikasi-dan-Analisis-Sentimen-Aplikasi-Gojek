import re
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_tokenizer(path):
    """Load tokenizer for LSTM model"""
    with open(path, "rb") as f:
        return pickle.load(f)

def clean_text(text):
    """Basic text cleaning for LSTM model"""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_text(text, tokenizer, maxlen=100):
    """Preprocess text for LSTM model"""
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
    return padded

def preprocess_text_sklearn(text):
    """Preprocess text for Random Forest model (more comprehensive cleaning)"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove extra whitespaces, newlines, and tabs
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation but keep some meaningful ones
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove numbers (optional, you might want to keep them)
    # text = re.sub(r'\d+', '', text)
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text.strip()

def highlight_keywords(text):
    """Highlight positive and negative keywords in text"""
    # Extended list of negative and positive words
    negative_words = [
        "jelek", "lambat", "parah", "buruk", "error", "tidak", "cancel", "gagal",
        "rusak", "lemot", "macet", "mahal", "kecewa", "susah", "ribet", "lama",
        "bermasalah", "salah", "hancur", "payah", "mengecewakan", "terburuk"
    ]
    
    positive_words = [
        "bagus", "cepat", "mantap", "baik", "terbaik", "senang", "puas",
        "hebat", "luar biasa", "memuaskan", "sempurna", "mudah", "lancar",
        "recommended", "top", "keren", "oke", "praktis", "efisien", "nyaman"
    ]

    # Highlight negative words with red background
    for word in negative_words:
        pattern = rf"\b({re.escape(word)})\b"
        text = re.sub(pattern, r"<mark style='background-color: #ffcccc; color: #d32f2f; padding: 2px 4px; border-radius: 3px;'>\1</mark>", text, flags=re.IGNORECASE)

    # Highlight positive words with green background
    for word in positive_words:
        pattern = rf"\b({re.escape(word)})\b"
        text = re.sub(pattern, r"<mark style='background-color: #ccffcc; color: #388e3c; padding: 2px 4px; border-radius: 3px;'>\1</mark>", text, flags=re.IGNORECASE)

    return text

def create_sample_models():
    """
    Function to create sample Random Forest model and TF-IDF vectorizer
    This is just for demonstration - you should train with real data
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pickle
    
    # Sample data (replace with your actual training data)
    sample_texts = [
        "aplikasi bagus dan cepat",
        "driver ramah dan profesional", 
        "pelayanan memuaskan sekali",
        "aplikasi lemot dan sering error",
        "driver tidak sopan",
        "pembayaran bermasalah",
        "biasa saja tidak istimewa",
        "standar aplikasi ojol pada umumnya"
    ]
    
    sample_labels = [2, 2, 2, 0, 0, 0, 1, 1]  # 0: negatif, 1: netral, 2: positif
    
    # Preprocess texts
    processed_texts = [preprocess_text_sklearn(text) for text in sample_texts]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(processed_texts)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, sample_labels)
    
    # Save models
    with open("random_forest_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    print("Sample Random Forest model and TF-IDF vectorizer created successfully!")
    return rf_model, vectorizer
