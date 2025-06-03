import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Set page config
st.set_page_config(
    page_title="Gojek Review Sentiment Analysis",
    page_icon=":mag:",  # Using a text emoji instead of Unicode emoji
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models and vectorizers
@st.cache_resource
def load_resources():
    # Load traditional ML model
    with open('best_rf_sentiment_model.pkl', 'rb') as f:
        trad_model = pickle.load(f)
    
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Load deep learning model
    dl_model = load_model('models/lstm_sentiment_model.keras')
    
    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    with open('models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    return trad_model, vectorizer, dl_model, tokenizer, label_encoder

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\\d+', '', text)
    # Remove extra spaces
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

# Function to predict sentiment using traditional ML model
def predict_traditional(text, model, vectorizer):
    # Preprocess the text
    processed_text = preprocess_text(text)
    # Vectorize the text
    text_vectorized = vectorizer.transform([processed_text])
    # Predict
    prediction = model.predict(text_vectorized)[0]
    # Get prediction probabilities (approximated for SVM)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(text_vectorized)[0]
    else:
        # For models like SVM that don't have predict_proba
        decision = model.decision_function(text_vectorized)
        proba = np.exp(decision) / np.sum(np.exp(decision), axis=1).reshape(-1, 1)
        proba = proba[0]
    
    return prediction, proba

# Function to predict sentiment using deep learning model
def predict_deep_learning(text, model, tokenizer, label_encoder, max_length=100):
    # Preprocess the text
    processed_text = preprocess_text(text)
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    # Predict
    prediction_proba = model.predict(padded)[0]
    prediction_idx = np.argmax(prediction_proba)
    prediction = label_encoder.inverse_transform([prediction_idx])[0]
    
    return prediction, prediction_proba

# Function to highlight influential words
def highlight_influential_words(text, model, vectorizer):
    # Preprocess the text
    processed_text = preprocess_text(text)
    # Tokenize
    words = processed_text.split()
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Get coefficients from the model (if SVM)
    if hasattr(model, 'coef_'):
        coefficients = model.coef_
    else:
        # For models without coefficients, return empty highlights
        return []
    
    # Vectorize each word separately
    word_influences = []
    for word in words:
        if word in feature_names:
            idx = np.where(feature_names == word)[0][0]
            # Get the coefficient for each class
            influences = [coef[idx] for coef in coefficients]
            max_influence = max(influences)
            word_influences.append((word, max_influence))
    
    # Sort by absolute influence
    word_influences.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return word_influences[:10]  # Return top 10 influential words

# Main app function
def main():
    # Load resources
    trad_model, vectorizer, dl_model, tokenizer, label_encoder = load_resources()
    
    # App title
    st.title("Gojek Review Sentiment Analysis")
    st.markdown("### Analyze sentiment of Gojek app reviews")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This app analyzes the sentiment of Gojek app reviews using both "
        "traditional machine learning and deep learning models."
    )
    st.sidebar.header("Models")
    model_choice = st.sidebar.radio(
        "Choose a model:",
        ["Traditional ML (SVM)", "Deep Learning (BiLSTM)"]
    )
    
    # Input area
    st.subheader("Enter a review:")
    user_input = st.text_area("", "Aplikasi gojek sangat membantu dan mudah digunakan.", height=100)
    
    # Example reviews
    st.subheader("Or try an example:")
    examples = [
        "Aplikasi bagus dan sangat membantu",
        "Saya kecewa dengan layanan driver yang tidak ramah",
        "Lumayan, tapi masih perlu ditingkatkan",
        "Aplikasi sering error dan tidak bisa dibuka",
        "Terima kasih Gojek, sangat memudahkan hidup saya!"
    ]
    example_choice = st.selectbox("Select an example review:", examples)
    if st.button("Use Example"):
        user_input = example_choice
        st.rerun()
    
    # Add a predict button
    predict_button = st.button("Analyze Sentiment")
    
    if predict_button or user_input:
        # Show a spinner while processing
        with st.spinner("Analyzing sentiment..."):
            if model_choice == "Traditional ML (SVM)":
                prediction, proba = predict_traditional(user_input, trad_model, vectorizer)
                influential_words = highlight_influential_words(user_input, trad_model, vectorizer)
            else:
                prediction, proba = predict_deep_learning(user_input, dl_model, tokenizer, label_encoder)
                influential_words = []  # Not available for deep learning model
        
        # Display results
        st.subheader("Results")
        
        # Create columns for results
        col1, col2 = st.columns(2)
        
        with col1:
            # Show prediction
            sentiment_color = {
                "positif": "green",
                "netral": "blue",
                "negatif": "red"
            }
            st.markdown(f"### Sentiment: <span style='color:{sentiment_color[prediction]}'>{prediction.upper()}</span>", unsafe_allow_html=True)
            
            # Show confidence
            if model_choice == "Traditional ML (SVM)":
                st.text("Confidence scores (estimated):")
                for i, label in enumerate(trad_model.classes_):
                    st.progress(float(proba[i]) if len(proba) > 1 else float(proba))
                    st.text(f"{label}: {float(proba[i])*100 if len(proba) > 1 else float(proba)*100:.2f}%")
            else:
                st.text("Confidence scores:")
                for i, label in enumerate(label_encoder.classes_):
                    st.progress(float(proba[i]))
                    st.text(f"{label}: {float(proba[i])*100:.2f}%")
        
        with col2:
            # Display influential words for traditional ML
            if model_choice == "Traditional ML (SVM)" and influential_words:
                st.subheader("Influential Words")
                
                # Prepare data for bar chart
                words = [w[0] for w in influential_words]
                scores = [w[1] for w in influential_words]
                
                # Create a bar chart of influential words
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = ['green' if s > 0 else 'red' for s in scores]
                ax.barh(words, scores, color=colors)
                ax.set_xlabel('Influence Score')
                ax.set_title('Words Influencing the Sentiment')
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # Display the chart
                st.pyplot(fig)
                
                # Show highlighted text
                st.subheader("Highlighted Review")
                highlighted_text = user_input
                for word, score in influential_words:
                    if score > 0:  # Positive sentiment
                        highlighted_text = highlighted_text.replace(word, f"<span style='background-color:rgba(0,255,0,0.3)'>{word}</span>")
                    else:  # Negative sentiment
                        highlighted_text = highlighted_text.replace(word, f"<span style='background-color:rgba(255,0,0,0.3)'>{word}</span>")
                
                st.markdown(highlighted_text, unsafe_allow_html=True)
            else:
                st.subheader("Model Information")
                if model_choice == "Traditional ML (SVM)":
                    st.write("Model: Support Vector Machine")
                    st.write("Features: TF-IDF")
                else:
                    st.write("Model: Bidirectional LSTM")
                    st.write("Features: Word Embeddings (FastText)")
                    st.write("Note: Word influence visualization is not available for deep learning models.")
    
    # Footer
    st.markdown("---")
    st.markdown("© 2023 Gojek Sentiment Analysis Project")

if __name__ == "__main__":
    main()
