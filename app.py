import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import preprocess_text, load_tokenizer, highlight_keywords

MAXLEN = 100
labels = ["Negatif", "Netral", "Positif"]

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("lstm_sentiment_model.h5")
    tokenizer = load_tokenizer("tokenizer.pickle")
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# --- Sidebar ---
with st.sidebar:
    st.title("📘 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini digunakan untuk **analisis sentimen** terhadap ulasan pengguna aplikasi **Gojek**.

    Model ini dibangun menggunakan algoritma **LSTM (Long Short-Term Memory)**, salah satu metode Deep Learning yang unggul dalam memahami teks dan urutan kata.

    Masukkan ulasan, dan sistem akan mengklasifikasikannya sebagai **positif**, **netral**, atau **negatif**.
    """)

# --- Header utama ---
st.markdown("<h1 style='color: #4CAF50;'>Analisis Sentimen Ulasan Gojek 🚖</h1>", unsafe_allow_html=True)
st.write("Masukkan ulasan pengguna di bawah ini dan lihat hasil prediksi sentimennya secara langsung.")

# --- Input pengguna ---
with st.container():
    text_input = st.text_area("📝 Masukkan Ulasan Pengguna", height=150, placeholder="Contoh: Aplikasinya sangat membantu, driver datang tepat waktu...")

    if st.button("🔍 Analisis Sekarang"):
        if text_input.strip():
            sequence = preprocess_text(text_input, tokenizer, maxlen=MAXLEN)
            prediction = model.predict(sequence)[0]
            label_index = np.argmax(prediction)
            confidence = prediction[label_index]

            st.markdown("---")
            col1, col2 = st.columns([1.2, 1])

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
