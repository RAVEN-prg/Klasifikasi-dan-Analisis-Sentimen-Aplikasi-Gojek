import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from utils import preprocess_text_sklearn

def load_data(file_path):
    """
    Load dataset from CSV file
    Expected columns: 'text' (review text) and 'label' (0=negatif, 1=netral, 2=positif)
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"File {file_path} tidak ditemukan!")
        return None

def train_random_forest_model(df, test_size=0.2, random_state=42):
    """
    Train Random Forest model for sentiment analysis
    """
    print("🚀 Memulai training Random Forest model...")
    
    # Preprocess text data
    print("📝 Preprocessing text data...")
    df['processed_text'] = df['text'].apply(preprocess_text_sklearn)
    
    # Prepare features and labels
    X_text = df['processed_text'].values
    y = df['label'].values
    
    # Create TF-IDF features
    print("🔤 Membuat TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),  # unigram, bigram, trigram
        min_df=2,
        max_df=0.95,
        stop_words=None  # You might want to add Indonesian stop words
    )
    
    X_tfidf = vectorizer.fit_transform(X_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"📊 Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    
    # Train Random Forest
    print("🌳 Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    print("📈 Evaluating model...")
    train_score = rf_model.score(X_train, y_train)
    test_score = rf_model.score(X_test, y_test)
    
    print(f"Training Accuracy: {train_score:.4f}")
    print(f"Testing Accuracy: {test_score:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Detailed evaluation
    y_pred = rf_model.predict(X_test)
    labels = ["Negatif", "Netral", "Positif"]
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))
    
    print("\n🔍 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    print("\n🔍 Top 20 Most Important Features:")
    feature_names = vectorizer.get_feature_names_out()
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(min(20, len(feature_names))):
        idx = indices[i]
        print(f"{i+1:2d}. {feature_names[idx]:20s} ({importances[idx]:.4f})")
    
    return rf_model, vectorizer

def save_models(rf_model, vectorizer):
    """Save trained model and vectorizer with better compatibility"""
    print("\n💾 Saving models...")
    
    try:
        # Save with protocol 2 for better compatibility across Python versions
        with open("random_forest_model.pkl", "wb") as f:
            pickle.dump(rf_model, f, protocol=2)
        print("✅ Random Forest model saved as 'random_forest_model.pkl'")
        
        with open("tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f, protocol=2)
        print("✅ TF-IDF vectorizer saved as 'tfidf_vectorizer.pkl'")
        
        # Also save as joblib for additional compatibility
        try:
            import joblib
            joblib.dump(rf_model, "random_forest_model.joblib")
            joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
            print("✅ Alternative joblib files also saved")
        except ImportError:
            print("⚠️ joblib not available, only pickle files saved")
            
    except Exception as e:
        print(f"❌ Error saving models: {e}")
        raise

def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    sample_data = {
        'text': [
            # Positive reviews
            "Aplikasi Gojek sangat membantu, driver cepat datang dan pelayanan memuaskan",
            "Driver ramah dan profesional, aplikasi mudah digunakan",
            "Pelayanan terbaik, cepat dan efisien",
            "Sangat puas dengan layanan Gojek, recommended banget",
            "Aplikasi bagus, fitur lengkap dan praktis",
            "Driver sopan dan kendaraan bersih",
            "Pelayanan cepat dan harga terjangkau",
            "Gojek memudahkan mobilitas sehari-hari",
            
            # Negative reviews  
            "Aplikasi sering error dan lemot",
            "Driver tidak sopan dan terlambat datang",
            "Pelayanan mengecewakan, aplikasi bermasalah",
            "Harga terlalu mahal dan aplikasi susah digunakan",
            "Driver kasar dan kendaraan kotor",
            "Aplikasi crash terus, sangat mengganggu",
            "Pelayanan terburuk yang pernah ada",
            "Gojek semakin tidak berkualitas",
            
            # Neutral reviews
            "Aplikasi biasa saja, tidak istimewa",
            "Pelayanan standar seperti ojol pada umumnya",
            "Cukup bagus tapi masih bisa diperbaiki",
            "Tidak ada yang spesial dari aplikasi ini",
            "Pelayanan oke, harga standar",
            "Aplikasi lumayan, ada plus minusnya",
            "Gojek seperti aplikasi ojol lainnya",
            "Biasa aja, tidak buruk tapi tidak excellent juga"
        ],
        'label': [
            # Positive (2)
            2, 2, 2, 2, 2, 2, 2, 2,
            # Negative (0)  
            0, 0, 0, 0, 0, 0, 0, 0,
            # Neutral (1)
            1, 1, 1, 1, 1, 1, 1, 1
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv("sample_gojek_reviews.csv", index=False)
    print("✅ Sample dataset created as 'sample_gojek_reviews.csv'")
    return df

def main():
    """Main function to train Random Forest model"""
    print("🤖 Random Forest Sentiment Analysis Training Script")
    print("=" * 50)
    
    # Try to load existing dataset
    df = load_data("gojek_reviews.csv")  # Replace with your actual dataset file
    
    if df is None:
        print("📝 Creating sample dataset for demonstration...")
        df = create_sample_dataset()
        print(f"📊 Sample dataset created with {len(df)} reviews")
    else:
        print(f"📊 Dataset loaded with {len(df)} reviews")
        print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Train model
    rf_model, vectorizer = train_random_forest_model(df)
    
    # Save models
    save_models(rf_model, vectorizer)
    
    print("\n🎉 Training completed successfully!")
    print("You can now use the Random Forest model in your Streamlit app.")

if __name__ == "__main__":
    main()
