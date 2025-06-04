# Gojek Sentiment Analysis App 🚖

Aplikasi analisis sentimen berbasis web untuk menganalisis ulasan pengguna aplikasi Gojek menggunakan Streamlit. Aplikasi ini menyediakan dua model machine learning yang berbeda: **LSTM** dan **Random Forest** dengan kemampuan interpretasi kata kunci.

## 🌟 Fitur Utama

- **Dual Model Analysis**: Pilihan antara model LSTM dan Random Forest
- **Real-time Prediction**: Analisis sentimen secara langsung
- **Keyword Highlighting**: Penyorotan kata kunci penting (khusus Random Forest)
- **Interactive Visualization**: Grafik confidence score untuk setiap prediksi
- **User-friendly Interface**: Interface yang mudah digunakan dengan Streamlit

## 📊 Model yang Tersedia

### 1. LSTM (Long Short-Term Memory)
- **Pendekatan**: Deep Learning
- **Keunggulan**:
  - Memahami konteks dan urutan kata dalam teks
  - Mampu menangkap hubungan jarak jauh antara kata-kata
  - Performa baik untuk teks yang kompleks

### 2. Random Forest
- **Pendekatan**: Ensemble Learning
- **Keunggulan**:
  - Menyoroti kata kunci penting yang mempengaruhi prediksi
  - Interpretasi fitur-fitur penting untuk setiap kelas sentimen
  - Proses prediksi yang lebih cepat
  - Visual highlighting untuk kata positif dan negatif

## 🎯 Klasifikasi Sentimen

Aplikasi mengklasifikasikan ulasan ke dalam 3 kategori:
- **Negatif** (0): Sentimen negatif
- **Netral** (1): Sentimen netral
- **Positif** (2): Sentimen positif

## 🛠️ Instalasi dan Setup

### Prerequisites
```bash
Python 3.7+
```

### Instalasi Dependencies
```bash
pip install -r requirements.txt
```

### Requirements.txt
```
streamlit>=1.28.0
tensorflow>=2.12.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
joblib>=1.1.0
```

### Struktur Direktori
```
├── app.py                          # Aplikasi utama Streamlit
├── utils.py                        # Fungsi utility
├── requirements.txt                # Dependencies
├── README.md                       # Dokumentasi
├── Notebook
│    ├── Preprocessing_Analisis_Senitmen.ipynb   # Melakukan perpocessing data mentah
│    ├── Train_RandomForest_Model.ipynb          # melakukan training dengan model random forest
│    ├── Training_LSTM_Model.ipynb               # melakukan training dengan model LSTM
├── Dataset
│    ├── GojekAppReview_1.csv         # data mentah 
│    ├── processed_gojek_reviews.csv  # data preprocessed
│    ├── cleaned_data.csv             # data bersih dengan dua kolom 
├── model/
│   ├── lstm_sentiment_model.h5     # Model LSTM
│   ├── rf_sentiment_model.pkl      # Model Random Forest
│   └── tfidf_vectorizer.pkl        # TF-IDF Vectorizer
└── tokenizer.pickle               # Tokenizer untuk LSTM
```

## 🚀 Cara Menjalankan

1. **Clone repository** (jika menggunakan Git):
```bash
git clone <repository-url>
cd gojek-sentiment-analysis
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Jalankan aplikasi**:
```bash
streamlit run app.py
```

4. **Akses aplikasi** di browser: `http://localhost:8501`

## 📝 Cara Penggunaan

1. **Pilih Model**: Gunakan sidebar untuk memilih antara "Random Forest" atau "LSTM"
2. **Input Teks**: Masukkan ulasan pengguna di text area
3. **Analisis**: Klik tombol "🔍 Analisis Sekarang"
4. **Lihat Hasil**: 
   - Prediksi sentimen dengan confidence score
   - Grafik visualisasi confidence untuk setiap kelas
   - Kata kunci yang disorot (untuk Random Forest)

## 🔧 Komponen Utama

### app.py
File utama yang berisi:
- Interface Streamlit
- Loading dan caching model
- Logika prediksi untuk kedua model
- Visualisasi hasil

### utils.py
Berisi fungsi-fungsi utility:
- `load_tokenizer()`: Memuat tokenizer untuk LSTM
- `clean_text()`: Preprocessing teks
- `preprocess_text()`: Preprocessing khusus untuk LSTM
- `highlight_keywords_rf()`: Highlighting kata kunci untuk Random Forest

## 📈 Interpretasi Hasil

### Confidence Score
- Nilai antara 0-1 yang menunjukkan tingkat kepercayaan model
- Semakin tinggi nilai, semakin yakin model dengan prediksinya

### Keyword Highlighting (Random Forest)
- **Hijau**: Kata-kata yang berkontribusi pada sentimen positif
- **Merah**: Kata-kata yang berkontribusi pada sentimen negatif

## ⚙️ Konfigurasi

### Parameter yang dapat disesuaikan:
- `MAXLEN = 100`: Panjang maksimum sequence untuk LSTM
- `top_n = 10`: Jumlah fitur teratas untuk highlighting

## 🧪 Contoh Penggunaan

**Input Positif:**
```
"Aplikasi Gojek sangat membantu, driver datang cepat dan pelayanan memuaskan!"
```

**Input Negatif:**
```
"Aplikasi sering error, driver lama datang, pelayanan mengecewakan"
```

**Input Netral:**
```
"Aplikasi Gojek biasa saja, tidak ada yang istimewa"
```

## 🔍 Troubleshooting

### Model tidak dapat dimuat
- Pastikan file model ada di direktori `model/`
- Periksa apakah `tokenizer.pickle` ada di root directory

### Error saat preprocessing
- Pastikan input tidak kosong
- Periksa format encoding teks input

### Streamlit tidak dapat diakses
- Pastikan port 8501 tidak digunakan aplikasi lain
- Coba jalankan dengan port berbeda: `streamlit run app.py --server.port 8502`

## 📊 Performa Model

Model telah dilatih dan dioptimalkan untuk:
- Akurasi tinggi dalam klasifikasi sentimen ulasan Gojek
- Kecepatan prediksi real-time
- Interpretabilitas hasil (khusus Random Forest)

## 🤝 Kontribusi

Untuk berkontribusi pada proyek ini:
1. Fork repository
2. Buat branch fitur baru
3. Commit perubahan
4. Push ke branch
5. Buat Pull Request

## 📄 Lisensi

Proyek ini menggunakan lisensi [MIT License](LICENSE).

## 👨‍💻 Author

Dibuat dengan ❤️ untuk analisis sentimen ulasan Gojek

---

**Catatan**: Pastikan semua file model dan dependencies telah terinstall sebelum menjalankan aplikasi.
