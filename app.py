import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

from utils import preprocess_text, load_tokenizer, highlight_keywords_rf, clean_text

MAXLEN = 100
labels = ["Negatif", "Netral", "Positif"]

@st.cache_resource(show_spinner=False)
def load_lstm_model():
    model = tf.keras.models.load_model("model/lstm_sentiment_model.h5")
    tokenizer = load_tokenizer("tokenizer.pickle")
    return model, tokenizer

@st.cache_resource(show_spinner=False)
def load_rf_model():
    model = joblib.load("model/rf_sentiment_model.pkl")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
    # Get top features for each class
    feature_names = vectorizer.get_feature_names_out()
    top_features = {
        'positive': get_top_features(model, feature_names, class_idx=2),
        'negative': get_top_features(model, feature_names, class_idx=0)
    }
    return model, vectorizer, top_features

def get_top_features(model, feature_names, class_idx, top_n=10):
    """Get top n features for a given class"""
    importances = model.feature_importances_
    if len(importances.shape) > 1:  # For multi-class
        importances = importances[class_idx]
    top_indices = np.argsort(importances)[-top_n:][::-1]
    return [feature_names[i] for i in top_indices]

# --- Sidebar ---
with st.sidebar:
    st.title("📘 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini digunakan untuk **analisis sentimen** terhadap ulasan pengguna aplikasi **Gojek**.

    Model utama menggunakan **Random Forest** dengan kemampuan interpretasi kata kunci penting.
    """)

    model_option = st.selectbox("🔍 Pilih Model Analisis", ["Random Forest", "LSTM"])

# --- Header utama ---
st.markdown("<h1 style='color: #4CAF50;'>Analisis Sentimen Ulasan Gojek 🚖</h1>", unsafe_allow_html=True)

# Deskripsi model yang dinamis
if model_option == "Random Forest":
    model_description = """
    **Model Random Forest** digunakan untuk analisis sentimen dengan kemampuan interpretasi yang baik.
    Keunggulan:
    - Menyoroti kata kunci penting yang mempengaruhi hasil prediksi
    - Dapat menunjukkan fitur-fitur penting untuk setiap kelas sentimen
    - Lebih cepat dalam proses prediksi
    """
else:
    model_description = """
    **Model LSTM** (Long Short-Term Memory) digunakan untuk analisis sentimen dengan pendekatan deep learning.
    Keunggulan:
    - Memahami konteks dan urutan kata dalam teks
    - Mampu menangkap hubungan jarak jauh antara kata-kata
    - Performa baik untuk teks yang kompleks
    """

st.write(f"""
Masukkan ulasan pengguna di bawah ini dan lihat hasil prediksi sentimennya secara langsung. 
Saat ini anda menggunakan **{model_option}** sebagai model analisis.

{model_description}
""")

# --- Load model ---
# Load Random Forest by default
rf_model, tfidf_vectorizer, top_features = load_rf_model()

if model_option == "LSTM":
    model, tokenizer = load_lstm_model()

# --- Input pengguna ---
text_input = st.text_area("📝 Masukkan Ulasan Pengguna", height=150, placeholder="Contoh: Aplikasinya sangat membantu, driver datang tepat waktu...")

if st.button("🔍 Analisis Sekarang"):
    if text_input.strip():
        st.markdown("---")
        col1, col2 = st.columns([1.2, 1])

        if model_option == "LSTM":
            sequence = preprocess_text(text_input, tokenizer, maxlen=MAXLEN)
            prediction = model.predict(sequence)[0]
            highlighted_text = text_input  # No highlighting for LSTM
        else:
            clean = clean_text(text_input)
            vector = tfidf_vectorizer.transform([clean])
            prediction = rf_model.predict_proba(vector)[0]
            # Highlight keywords based on Random Forest's important features
            highlighted_text = highlight_keywords_rf(text_input, top_features)

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

        if model_option == "Random Forest":
            st.markdown("### ✨ Kata Kunci yang Disorot")
            st.markdown(highlighted_text, unsafe_allow_html=True)
        
    else:
        st.warning("Silakan masukkan teks ulasan terlebih dahulu.") 
