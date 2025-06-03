import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

from utils import preprocess_text, load_tokenizer, highlight_keywords, clean_text

MAXLEN = 100
labels = ["Negatif", "Netral", "Positif"]

@st.cache_resource(show_spinner=False)
def load_lstm_model():
    model = tf.keras.models.load_model("lstm_sentiment_model.h5")
    tokenizer = load_tokenizer("tokenizer.pickle")
    return model, tokenizer

@st.cache_resource(show_spinner=False)
def load_rf_model():
    model = joblib.load("rf_sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

# --- Sidebar ---
with st.sidebar:
    st.title("📘 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini digunakan untuk **analisis sentimen** terhadap ulasan pengguna aplikasi **Gojek**.

    Pilih model yang ingin digunakan: **LSTM** (Deep Learning) atau **Random Forest** (Machine Learning).
    """)

    model_option = st.selectbox("🔍 Pilih Model Analisis", ["LSTM", "Random Forest"])

# --- Header utama ---
st.markdown("<h1 style='color: #4CAF50;'>Analisis Sentimen Ulasan Gojek 🚖</h1>", unsafe_allow_html=True)
st.write("Masukkan ulasan pengguna di bawah ini dan lihat hasil prediksi sentimennya secara langsung.")

# --- Load model sesuai pilihan ---
if model_option == "LSTM":
    model, tokenizer = load_lstm_model()
else:
    rf_model, tfidf_vectorizer = load_rf_model()

# --- Input pengguna ---
text_input = st.text_area("📝 Masukkan Ulasan Pengguna", height=150, placeholder="Contoh: Aplikasinya sangat membantu, driver datang tepat waktu...")

if st.button("🔍 Analisis Sekarang"):
    if text_input.strip():
        st.markdown("---")
        col1, col2 = st.columns([1.2, 1])

        if model_option == "LSTM":
            sequence = preprocess_text(text_input, tokenizer, maxlen=MAXLEN)
            prediction = model.predict(sequence)[0]
        else:
            clean = clean_text(text_input)
            vector = tfidf_vectorizer.transform([clean])
            prediction = rf_model.predict_proba(vector)[0]

        label_index = np.argmax(prediction)
        confidence = prediction[label_index]

        with col1:
            st.subheader("📊 Hasil Prediksi")
            st.success(f"**Sentimen:** {labels[label_index]}")
            st.write(f"**Confidence Score:** {confidence:.2f}")

        with col2:
            fig, ax = plt.subplots()
            ax.bar(labels, prediction, color=["#f44336", "#9e9e9e", "#4CAF50"])
            ax.set_ylabel("Confidence")
            ax.set_ylim(0, 1)
            ax.set_title("Confidence Score")
            st.pyplot(fig)

        st.markdown("### ✨ Kata Kunci yang Disorot")
        highlighted_text = highlight_keywords(text_input)
        st.markdown(highlighted_text, unsafe_allow_html=True)
    else:
        st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
