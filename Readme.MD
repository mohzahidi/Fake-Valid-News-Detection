#  Fake & Valid News Detection System

## 📋 Deskripsi Proyek

Proyek ini merupakan sistem deteksi berita palsu (fake news) dan berita valid menggunakan pendekatan **Natural Language Processing (NLP)** dan **Deep Learning**. Sistem ini membandingkan performa tiga model machine learning/deep learning untuk mengklasifikasikan berita ke dalam dua kategori: **Fake** atau **Valid**.

###  Tujuan
- Mengembangkan sistem klasifikasi berita palsu yang akurat
- Membandingkan performa tiga model berbeda
- Menyediakan antarmuka web yang user-friendly untuk deteksi berita

###  Fitur Utama
- Deteksi berita palsu secara real-time
- Perbandingan hasil dari 3 model berbeda
- Visualisasi confidence score
- Antarmuka web yang responsif

---

##  Dataset dan Preprocessing

### Dataset
| Informasi | Detail |
|-----------|--------|
| **Sumber** | https://www.kaggle.com/datasets/hassanamin/textdb3 |
| **Jumlah Data** | [6,335 News] |
| **Distribusi** | Fake: [X] berita, Valid: [Y] berita |
| **Bahasa** | [English] |

### Preprocessing Pipeline
Raw Text → Cleaning → Tokenization → Normalization → Vectorization → Model Input

**Tahapan Preprocessing:**

1. **Text Cleaning**
   - Menghapus karakter khusus dan angka
   - Menghapus URL dan mention
   - Menghapus HTML tags
   - Konversi ke lowercase

2. **Tokenization**
   - Memecah teks menjadi token/kata
   - Menggunakan tokenizer sesuai model (BERT Tokenizer, dll.)

3. **Normalization**
   - Stopword removal
   - Lemmatization/Stemming
   - Handling kata tidak baku

4. **Vectorization**
   - TF-IDF untuk model klasik
   - Word Embedding untuk model transformer

## 🤖 Model yang Digunakan

### 1. Logistic Regression + TF-IDF

| Aspek | Detail |
|-------|--------|
| **Tipe** | Machine Learning Klasik |
| **Vectorizer** | TF-IDF (Term Frequency-Inverse Document Frequency) |
| **Hyperparameter** | C=1.0, max_iter=1000 |
| **Kelebihan** | Cepat, interpretable, ringan |
| **Kekurangan** | Tidak menangkap konteks semantik |

### 2. BERT (Bidirectional Encoder Representations from Transformers)

| Aspek | Detail |
|-------|--------|
| **Tipe** | Deep Learning - Transformer |
| **Pre-trained Model** | bert-base-uncased / indobert-base |
| **Max Length** | 512 tokens |
| **Fine-tuning** | 3-5 epochs |
| **Kelebihan** | Memahami konteks bidirectional, akurasi tinggi |
| **Kekurangan** | Berat, membutuhkan GPU |

### 3. DistilBERT

| Aspek | Detail |
|-------|--------|
| **Tipe** | Deep Learning - Distilled Transformer |
| **Pre-trained Model** | distilbert-base-uncased |
| **Max Length** | 512 tokens |
| **Fine-tuning** | 3-5 epochs |
| **Kelebihan** | 60% lebih cepat dari BERT, 97% performa BERT |
| **Kekurangan** | Sedikit kurang akurat dari BERT |

### 📊 Analisis Perbandingan

#### Performa
- **Model Terbaik**: DistilBERT dengan akurasi 97%
- **Trade-off**: DistilBERT memberikan keseimbangan terbaik antara akurasi dan kecepatan

#### Waktu Inferensi
| Model | Waktu Prediksi (per teks) |
|-------|---------------------------|
| Logistic Regression | 10-50 ms |
| BERT | 100-200 ms |
| DistilBERT | 200-400 ms |

#### Kesimpulan Analisis
1. **BERT** menunjukkan performa terbaik dalam menangkap nuansa semantik berita
2. **DistilBERT** menjadi pilihan optimal untuk deployment dengan resource terbatas
3. **Logistic Regression** cocok untuk baseline dan sistem dengan keterbatasan komputasi

---

##  Panduan Menjalankan Sistem Website Secara Lokal

### Prasyarat

- Python 3.8 atau lebih tinggi
- pip (Python package manager)
- Git
- RAM minimal 8GB (untuk model BERT)
- (Opsional) NVIDIA GPU untuk akselerasi

### Langkah-langkah Instalasi

#### 1. Clone Repository

```bash
git clone https://github.com/mohzahidi/Fake-Valid-News-Detection.git
cd Fake-Valid-News-Detection

2. Buat Virtual Environment
bashDownloadCopy code# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
bashDownloadCopy codepip install -r requirements.txt
4. Download Model (Jika Diperlukan)
bashDownloadCopy code# Jalankan script untuk download model pre-trained
python download_models.py
5. Jalankan Aplikasi
bashDownloadCopy code# Untuk Flask
python app.py

# Untuk Streamlit
streamlit run app.py
6. Akses Website
Buka browser dan akses:
http://localhost:5000

atau untuk Streamlit:
http://localhost:8501
