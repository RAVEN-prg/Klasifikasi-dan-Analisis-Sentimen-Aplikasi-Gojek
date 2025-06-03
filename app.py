import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import preprocess_text, load_tokenizer, highlight_keywords, preprocess_text_sklearn

MAXLEN = 100
labels = ["Negatif", "Netral", "Positif"]

@st.cache_resource(show_spinner=False)
def load_lstm_model_and_tokenizer():
    model = tf.keras.models.load_model("lstm_sentiment_model.h5")
    tokenizer = load_tokenizer("tokenizer.pickle")
    return model, tokenizer

@st.cache_resource(show_spinner=False)
def load_random_forest_model():
    try:
        with open("best_rf_sentiment_model.pkl", "rb") as f:
            rf_model = pickle.load(f)
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return rf_model, vectorizer
    except FileNotFoundError:
        st.error("Model Random Forest tidak ditemukan. Pastikan file 'random_forest_model.pkl' dan 'tfidf_vectorizer.pkl' tersedia.")
        return None, None

# Load models
lstm_model, lstm_tokenizer = load_lstm_model_and_tokenizer()
rf_model, tfidf_vectorizer = load_random_forest_model()

# --- Sidebar ---
with st.sidebar:
    st.title("📘 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini digunakan untuk **analisis sentimen** terhadap ulasan pengguna aplikasi **Gojek**.

    **Model yang tersedia:**
    - **LSTM (Long Short-Term Memory)**: Model Deep Learning yang unggul dalam memahami konteks dan urutan kata
    - **Random Forest**: Model Machine Learning ensemble yang cepat dan efektif untuk klasifikasi teks

    Masukkan ulasan, dan sistem akan mengklasifikasikannya sebagai **positif**, **netral**, atau **negatif**.
    """)
    
    st.markdown("---")
    
    # Model selection
    st.subheader("🔧 Pilih Model")
    selected_model = st.selectbox(
        "Pilih model untuk analisis sentimen:",
        ["LSTM", "Random Forest"],
        help="LSTM lebih baik untuk konteks, Random Forest lebih cepat"
    )

# --- Header utama ---
st.markdown("<h1 style='color: #4CAF50;'>Analisis Sentimen Ulasan Gojek 🚖</h1>", unsafe_allow_html=True)
st.write("Masukkan ulasan pengguna di bawah ini dan lihat hasil prediksi sentimennya secara langsung.")

# Display selected model info
if selected_model == "LSTM":
    st.info("🧠 **Model LSTM** dipilih - Analisis mendalam dengan pemahaman konteks")
else:
    st.info("🌳 **Model Random Forest** dipilih - Analisis cepat dengan fitur statistik")

# --- Input pengguna ---
with st.container():
    text_input = st.text_area("📝 Masukkan Ulasan Pengguna", height=150, placeholder="Contoh: Aplikasinya sangat membantu, driver datang tepat waktu...")

    if st.button("🔍 Analisis Sekarang"):
        if text_input.strip():
            if selected_model == "LSTM":
                # LSTM prediction
                sequence = preprocess_text(text_input, lstm_tokenizer, maxlen=MAXLEN)
                prediction = lstm_model.predict(sequence)[0]
                label_index = np.argmax(prediction)
                confidence = prediction[label_index]
                
            else:  # Random Forest
                if rf_model is not None and tfidf_vectorizer is not None:
                    # Random Forest prediction
                    processed_text = preprocess_text_sklearn(text_input)
                    text_vector = tfidf_vectorizer.transform([processed_text])
                    prediction_proba = rf_model.predict_proba(text_vector)[0]
                    label_index = np.argmax(prediction_proba)
                    confidence = prediction_proba[label_index]
                    prediction = prediction_proba
                else:
                    st.error("Model Random Forest tidak dapat dimuat. Menggunakan model LSTM sebagai fallback.")
                    sequence = preprocess_text(text_input, lstm_tokenizer, maxlen=MAXLEN)
                    prediction = lstm_model.predict(sequence)[0]
                    label_index = np.argmax(prediction)
                    confidence = prediction[label_index]

            st.markdown("---")
            col1, col2 = st.columns([1.2, 1])

            with col1:
                st.subheader("📊 Hasil Prediksi")
                
                # Color coding for sentiment
                if label_index == 0:  # Negatif
                    st.error(f"**Sentimen:** {labels[label_index]}")
                elif label_index == 1:  # Netral
                    st.warning(f"**Sentimen:** {labels[label_index]}")
                else:  # Positif
                    st.success(f"**Sentimen:** {labels[label_index]}")
                
                st.write(f"**Confidence Score:** {confidence:.2f}")
                st.write(f"**Model yang digunakan:** {selected_model}")

            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ["#f44336", "#ff9800", "#4CAF50"]
                bars = ax.bar(labels, prediction, color=colors)
                ax.set_ylabel("Confidence Score")
                ax.set_ylim(0, 1)
                ax.set_title(f"Distribusi Confidence - Model {selected_model}")
                
                # Add value labels on bars
                for bar, value in zip(bars, prediction):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)

            st.markdown("### ✨ Kata Kunci yang Disorot")
            highlighted_text = highlight_keywords(text_input)
            st.markdown(highlighted_text, unsafe_allow_html=True)
            
            # Additional model comparison info
            with st.expander("ℹ️ Informasi Model"):
                if selected_model == "LSTM":
                    st.markdown("""
                    **LSTM (Long Short-Term Memory)**
                    - ✅ Memahami konteks dan urutan kata dengan baik
                    - ✅ Dapat menangkap nuansa bahasa yang kompleks
                    - ⚠️ Membutuhkan waktu prediksi lebih lama
                    - ⚠️ Memerlukan preprocessing khusus (tokenisasi)
                    """)
                else:
                    st.markdown("""
                    **Random Forest**
                    - ✅ Prediksi sangat cepat
                    - ✅ Tidak memerlukan GPU
                    - ✅ Interpretable dan robust
                    - ⚠️ Kurang memahami konteks urutan kata
                    - ⚠️ Bergantung pada fitur TF-IDF
                    """)
        else:
            st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
