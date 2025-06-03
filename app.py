import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import preprocess_text, load_tokenizer, highlight_keywords

MAXLEN = 100
labels = ["Negatif", "Netral", "Positif"]

@st.cache_resource(show_spinner=False)
def load_lstm_model():
    model = tf.keras.models.load_model("lstm_sentiment_model.h5")
    tokenizer = load_tokenizer("tokenizer.pickle")
    return model, tokenizer

@st.cache_resource(show_spinner=False)
def load_random_forest_model():
    try:
        with open("best_rf_sentiment_model.pkl", "rb") as f:
            rf_model = pickle.load(f)
        with open("tfidf_vectorizer (2).pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return rf_model, vectorizer
    except FileNotFoundError:
        st.error("File model Random Forest tidak ditemukan. Pastikan file 'random_forest_model.pkl' dan 'tfidf_vectorizer.pkl' tersedia.")
        return None, None

# Load models
lstm_model, tokenizer = load_lstm_model()
rf_model, vectorizer = load_random_forest_model()

# --- Sidebar ---
with st.sidebar:
    st.title("📘 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini digunakan untuk **analisis sentimen** terhadap ulasan pengguna aplikasi **Gojek**.

    **Model yang tersedia:**
    - **LSTM**: Menggunakan Deep Learning untuk memahami konteks dan urutan kata
    - **Random Forest**: Menggunakan Machine Learning klasik dengan TF-IDF features

    Masukkan ulasan, dan sistem akan mengklasifikasikannya sebagai **positif**, **netral**, atau **negatif**.
    """)
    
    # Model selection
    st.markdown("---")
    st.subheader("🤖 Pilih Model")
    selected_model = st.radio(
        "Pilih algoritma untuk analisis sentimen:",
        ["LSTM (Deep Learning)", "Random Forest (Machine Learning)"],
        help="LSTM lebih baik untuk konteks, Random Forest lebih cepat dan interpretable"
    )

# --- Header utama ---
st.markdown("<h1 style='color: #4CAF50;'>Analisis Sentimen Ulasan Gojek 🚖</h1>", unsafe_allow_html=True)
st.write("Masukkan ulasan pengguna di bawah ini dan lihat hasil prediksi sentimennya secara langsung.")

# Display selected model info
model_info = {
    "LSTM (Deep Learning)": "🧠 Menggunakan neural network untuk memahami konteks dan urutan kata",
    "Random Forest (Machine Learning)": "🌳 Menggunakan ensemble learning dengan TF-IDF features"
}
st.info(f"**Model aktif:** {selected_model} - {model_info[selected_model]}")

# --- Input pengguna ---
with st.container():
    text_input = st.text_area("📝 Masukkan Ulasan Pengguna", height=150, placeholder="Contoh: Aplikasinya sangat membantu, driver datang tepat waktu...")

    if st.button("🔍 Analisis Sekarang"):
        if text_input.strip():
            try:
                if selected_model == "LSTM (Deep Learning)":
                    # LSTM prediction
                    sequence = preprocess_text(text_input, tokenizer, maxlen=MAXLEN)
                    prediction = lstm_model.predict(sequence, verbose=0)[0]
                    label_index = np.argmax(prediction)
                    confidence = prediction[label_index]
                    
                elif selected_model == "Random Forest (Machine Learning)":
                    if rf_model is None or vectorizer is None:
                        st.error("Model Random Forest tidak tersedia. Silakan pilih model LSTM.")
                        st.stop()
                    
                    # Random Forest prediction
                    from utils import clean_text
                    cleaned_text = clean_text(text_input)
                    text_vectorized = vectorizer.transform([cleaned_text])
                    
                    # Get prediction probabilities
                    prediction_proba = rf_model.predict_proba(text_vectorized)[0]
                    label_index = rf_model.predict(text_vectorized)[0]
                    confidence = prediction_proba[label_index]
                    prediction = prediction_proba

                st.markdown("---")
                col1, col2 = st.columns([1.2, 1])

                with col1:
                    st.subheader("📊 Hasil Prediksi")
                    
                    # Color coding for sentiment
                    sentiment_colors = {0: "🔴", 1: "🟡", 2: "🟢"}
                    st.success(f"**Sentimen:** {sentiment_colors[label_index]} {labels[label_index]}")
                    st.write(f"**Confidence Score:** {confidence:.2f}")
                    st.write(f"**Model yang digunakan:** {selected_model}")

                with col2:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = ["#f44336", "#ff9800", "#4CAF50"]
                    bars = ax.bar(labels, prediction, color=colors)
                    ax.set_ylabel("Confidence Score")
                    ax.set_ylim(0, 1)
                    ax.set_title(f"Distribusi Confidence - {selected_model}")
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, prediction):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    st.pyplot(fig)

                # Model comparison section
                st.markdown("### 🔄 Perbandingan Model")
                col3, col4 = st.columns(2)
                
                with col3:
                    st.write("**LSTM Prediction:**")
                    lstm_seq = preprocess_text(text_input, tokenizer, maxlen=MAXLEN)
                    lstm_pred = lstm_model.predict(lstm_seq, verbose=0)[0]
                    lstm_label = np.argmax(lstm_pred)
                    st.write(f"Sentimen: {labels[lstm_label]} ({lstm_pred[lstm_label]:.3f})")
                
                with col4:
                    if rf_model is not None and vectorizer is not None:
                        st.write("**Random Forest Prediction:**")
                        from utils import clean_text
                        rf_text = vectorizer.transform([clean_text(text_input)])
                        rf_pred = rf_model.predict_proba(rf_text)[0]
                        rf_label = rf_model.predict(rf_text)[0]
                        st.write(f"Sentimen: {labels[rf_label]} ({rf_pred[rf_label]:.3f})")
                    else:
                        st.write("Random Forest model tidak tersedia")

                st.markdown("### ✨ Kata Kunci yang Disorot")
                highlighted_text = highlight_keywords(text_input)
                st.markdown(highlighted_text, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi: {str(e)}")
        else:
            st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
