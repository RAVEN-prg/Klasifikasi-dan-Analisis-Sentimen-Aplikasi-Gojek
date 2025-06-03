import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle # Untuk memuat model RF
from utils import preprocess_text_lstm, preprocess_text_rf, load_tokenizer, load_vectorizer, highlight_keywords

MAXLEN = 100
labels = ["Negatif", "Netral", "Positif"] # Pastikan urutan ini sesuai dengan output model Anda

# --- Pemuatan Model dan Tokenizer/Vectorizer ---
@st.cache_resource(show_spinner="Memuat model LSTM...")
def load_lstm_model_and_tokenizer():
    model_lstm = tf.keras.models.load_model("lstm_sentiment_model.h5")
    tokenizer_lstm = load_tokenizer("tokenizer.pickle")
    return model_lstm, tokenizer_lstm

@st.cache_resource(show_spinner="Memuat model Random Forest...")
def load_rf_model_and_vectorizer():
    try:
        with open("best_rfsentiment_model.pkl", "rb") as f_model:
            model_rf = pickle.load(f_model)
        vectorizer_rf = load_vectorizer("tfidf_vectorizer.pkl")
        return model_rf, vectorizer_rf
    except FileNotFoundError:
        st.error("File model Random Forest (random_forest_model.pkl) atau vectorizer (rf_vectorizer.pkl) tidak ditemukan.")
        st.info("Pastikan Anda telah meletakkan file model dan vectorizer yang sudah dilatih di direktori yang sama dengan app.py, atau buat model dummy jika hanya ingin menjalankan struktur aplikasi.")
        return None, None # Kembalikan None jika file tidak ditemukan

model_lstm, tokenizer_lstm = load_lstm_model_and_tokenizer()
model_rf, vectorizer_rf = load_rf_model_and_vectorizer()

# --- Sidebar ---
with st.sidebar:
    st.title("📘 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini digunakan untuk **analisis sentimen** terhadap ulasan pengguna aplikasi **Gojek**.
    """)

    # Pilihan Model
    st.subheader("🤖 Pilih Model Analisis")
    selected_model_name = st.radio(
        "Model yang ingin digunakan:",
        ("LSTM", "Random Forest"),
        help="LSTM adalah model Deep Learning, sedangkan Random Forest adalah model Machine Learning klasik."
    )

    if selected_model_name == "LSTM":
        st.markdown("""
        Model **LSTM (Long Short-Term Memory)** adalah salah satu metode Deep Learning yang unggul dalam memahami teks dan urutan kata.
        """)
    elif selected_model_name == "Random Forest" and model_rf:
        st.markdown("""
        Model **Random Forest** adalah algoritma ensemble learning yang menggabungkan beberapa decision tree untuk meningkatkan akurasi dan mengurangi overfitting. Model ini umumnya lebih cepat dilatih daripada LSTM.
        """)
    elif selected_model_name == "Random Forest" and not model_rf:
        st.warning("Model Random Forest belum siap. Silakan periksa file model.")


    st.markdown("""
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
            prediction = None
            confidence_scores = None # Akan menjadi array probabilitas

            if selected_model_name == "LSTM":
                if model_lstm and tokenizer_lstm:
                    sequence = preprocess_text_lstm(text_input, tokenizer_lstm, maxlen=MAXLEN)
                    confidence_scores = model_lstm.predict(sequence)[0]
                else:
                    st.error("Model LSTM tidak berhasil dimuat.")
            
            elif selected_model_name == "Random Forest":
                if model_rf and vectorizer_rf:
                    processed_text_rf = preprocess_text_rf(text_input, vectorizer_rf)
                    try:
                        # predict_proba menghasilkan array probabilitas untuk setiap kelas
                        confidence_scores = model_rf.predict_proba(processed_text_rf)[0]
                        # Pastikan urutan kelas di 'labels' sesuai dengan output predict_proba model RF
                        # Jika model RF Anda mengeluarkan label dalam urutan berbeda, sesuaikan di sini
                        # Contoh: jika RF mengeluarkan [Netral, Negatif, Positif]
                        # dan labels = ["Negatif", "Netral", "Positif"]
                        # Anda mungkin perlu mengurutkan ulang confidence_scores agar sesuai dengan 'labels'
                        # Untuk sekarang, kita asumsikan urutannya sama.
                    except Exception as e:
                        st.error(f"Error saat prediksi dengan Random Forest: {e}")
                        st.error("Pastikan model Random Forest dan vectorizer Anda kompatibel dan telah dilatih dengan benar.")
                else:
                    st.error("Model Random Forest atau vectorizer tidak berhasil dimuat. Pastikan file .pkl ada.")

            if confidence_scores is not None and len(confidence_scores) == len(labels):
                label_index = np.argmax(confidence_scores)
                confidence = confidence_scores[label_index]

                st.markdown("---")
                col1, col2 = st.columns([1.2, 1])

                with col1:
                    st.subheader("📊 Hasil Prediksi")
                    # Atur warna berdasarkan sentimen
                    if labels[label_index] == "Positif":
                        st.success(f"**Sentimen:** {labels[label_index]}")
                    elif labels[label_index] == "Negatif":
                        st.error(f"**Sentimen:** {labels[label_index]}")
                    else: # Netral
                        st.info(f"**Sentimen:** {labels[label_index]}")
                    
                    st.write(f"**Confidence Score:** {confidence:.2f}")
                    st.write(f"*(Menggunakan model: {selected_model_name})*")

                with col2:
                    fig, ax = plt.subplots(figsize=(5,4)) # Ukuran disesuaikan agar lebih pas
                    # Warna diurutkan sesuai dengan labels: Negatif, Netral, Positif
                    bar_colors = []
                    for lbl in labels:
                        if lbl == "Negatif":
                            bar_colors.append("#f44336") # Merah
                        elif lbl == "Netral":
                            bar_colors.append("#9e9e9e") # Abu-abu
                        elif lbl == "Positif":
                            bar_colors.append("#4CAF50") # Hijau
                    
                    ax.bar(labels, confidence_scores, color=bar_colors)
                    ax.set_ylabel("Confidence")
                    ax.set_ylim(0, 1)
                    # ax.set_title("Confidence Score per Kategori") # Judul bisa dihilangkan jika terlalu ramai
                    # Putar label x-axis jika bertabrakan
                    plt.xticks(rotation=45, ha="right") 
                    plt.tight_layout() # Menyesuaikan layout agar tidak terpotong
                    st.pyplot(fig)

                st.markdown("### ✨ Kata Kunci yang Disorot")
                highlighted_text = highlight_keywords(text_input) # Fungsi ini tetap sama
                st.markdown(f"<div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>{highlighted_text}</div>", unsafe_allow_html=True)
            
            elif confidence_scores is None:
                # Pesan error spesifik sudah ditampilkan di atas jika model gagal dimuat
                pass
            else:
                st.error(f"Terjadi masalah dengan output probabilitas model. Jumlah kelas yang diprediksi ({len(confidence_scores)}) tidak sesuai dengan jumlah label ({len(labels)}).")

        else:
            st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
