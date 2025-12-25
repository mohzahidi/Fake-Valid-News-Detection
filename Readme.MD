#  Fake & Valid News Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

##  Deskripsi Proyek

Proyek ini merupakan sistem deteksi berita palsu (fake news) dan berita valid menggunakan pendekatan **Natural Language Processing (NLP)** dan **Deep Learning**. Sistem ini membandingkan performa tiga model machine learning/deep learning untuk mengklasifikasikan berita ke dalam dua kategori: **Fake** atau **Real**.

###  Tujuan
- Mengembangkan sistem klasifikasi berita palsu yang akurat
- Membandingkan performa tiga model berbeda (BERT, DistilBERT, LSTM)
- Menyediakan antarmuka web yang user-friendly untuk deteksi berita

###  Fitur Utama
-  Deteksi berita palsu secara real-time
-  Perbandingan hasil dari 3 model berbeda
-  Visualisasi confidence score dan probability distribution
-  Model consensus untuk hasil yang lebih reliable
-  Antarmuka web yang modern dan responsif
-  Support GPU acceleration

---

##  Dataset dan Preprocessing

### Dataset

| Informasi | Detail |
|-----------|--------|
| **Sumber** | [Kaggle - TextDB3](https://www.kaggle.com/datasets/hassanamin/textdb3) |
| **Jumlah Data** | 6,335 News |
| **Distribusi** | Fake: 3,164 berita, Real: 3,171 berita |
| **Bahasa** | English |
| **Split Ratio** | Train: 80%, Test: 20% |

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
   - Menggunakan tokenizer sesuai model (BERT Tokenizer, Keras Tokenizer)

3. **Normalization**
   - Stopword removal
   - Lemmatization/Stemming
   - Handling kata tidak baku

4. **Vectorization**
   - Word Embedding untuk model Transformer (BERT, DistilBERT)
   - Sequence padding untuk LSTM

---

##  Model yang Digunakan

### 1. BERT (Bidirectional Encoder Representations from Transformers)

| Aspek | Detail |
|-------|--------|
| **Tipe** | Deep Learning - Transformer |
| **Pre-trained Model** | `bert-base-uncased` |
| **Max Length** | 512 tokens |
| **Fine-tuning** | 3-5 epochs |
| **Kelebihan** | Memahami konteks bidirectional, akurasi tinggi |
| **Kekurangan** | Resource intensive, membutuhkan GPU |

### 2. DistilBERT

| Aspek | Detail |
|-------|--------|
| **Tipe** | Deep Learning - Distilled Transformer |
| **Pre-trained Model** | `distilbert-base-uncased` |
| **Max Length** | 512 tokens |
| **Fine-tuning** | 3-5 epochs |
| **Kelebihan** | 60% lebih cepat dari BERT, 97% performa BERT |
| **Kekurangan** | Sedikit trade-off pada kompleksitas tinggi |

### 3. LSTM (Long Short-Term Memory)

| Aspek | Detail |
|-------|--------|
| **Tipe** | Deep Learning - Recurrent Neural Network |
| **Architecture** | Embedding → LSTM → Dense |
| **Max Length** | 200 tokens |
| **Kelebihan** | Lebih ringan, cepat untuk training |
| **Kekurangan** | Kurang optimal untuk teks panjang |

---

##  Hasil Evaluasi Model

### BERT

| Metric | FAKE | REAL | Overall |
|--------|------|------|---------|
| **Precision** | 1.00 | 0.95 | - |
| **Recall** | 0.95 | 1.00 | - |
| **F1-Score** | 0.97 | 0.97 | - |
| **Accuracy** | - | - | **97.20%** |

<details>
<summary> Classification Report Detail</summary>


          precision    recall  f1-score   support

    FAKE       1.00      0.95      0.97       493
    REAL       0.95      1.00      0.97       507

accuracy                           0.97      1000

macro avg       0.97      0.97      0.97      1000
weighted avg       0.97      0.97      0.97      1000
</details>

### DistilBERT

| Metric | FAKE | REAL | Overall |
|--------|------|------|---------|
| **Precision** | 0.99 | 0.96 | - |
| **Recall** | 0.96 | 0.99 | - |
| **F1-Score** | 0.97 | 0.97 | - |
| **Accuracy** | - | - | **97.40%** |

<details>
<summary>📋 Classification Report Detail</summary>


          precision    recall  f1-score   support

    FAKE       0.99      0.96      0.97       493
    REAL       0.96      0.99      0.97       507

accuracy                           0.97      1000

macro avg       0.97      0.97      0.97      1000
weighted avg       0.97      0.97      0.97      1000
</details>

### LSTM

| Metric | FAKE | REAL | Overall |
|--------|------|------|---------|
| **Precision** | 0.96 | 0.89 | - |
| **Recall** | 0.88 | 0.96 | - |
| **F1-Score** | 0.92 | 0.92 | - |
| **Accuracy** | - | - | **91.95%** |

<details>
<summary> Classification Report Detail</summary>


          precision    recall  f1-score   support

    FAKE       0.96      0.88      0.92       633
    REAL       0.89      0.96      0.92       634

accuracy                           0.92      1267

macro avg       0.92      0.92      0.92      1267
weighted avg       0.92      0.92      0.92      1267
</details>

---

##  Perbandingan Model

### Ringkasan Performa

| Model | Accuracy | F1-Score (Avg) | Inference Time* |
|-------|----------|----------------|-----------------|
| **DistilBERT** 🥇 | 97.40% | 0.97 | ~50-100 ms |
| **BERT** 🥈 | 97.20% | 0.97 | ~100-200 ms |
| **LSTM** 🥉 | 91.95% | 0.92 | ~10-30 ms |

*Inference time per sample pada CPU

### Visualisasi Perbandingan


Accuracy Comparison:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DistilBERT  ████████████████████████████████████████ 97.40%
BERT        ███████████████████████████████████████░ 97.20%
LSTM        ████████████████████████████████░░░░░░░░ 91.95%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Analisis

| Aspek | Kesimpulan |
|-------|------------|
| **Model Terbaik** | DistilBERT dengan akurasi 97.40% |
| **Trade-off Terbaik** | DistilBERT - balance antara akurasi dan kecepatan |
| **Untuk Resource Terbatas** | LSTM - ringan dengan akurasi 91.95% |
| **Untuk Akurasi Maksimal** | BERT/DistilBERT - performa hampir identik |

### Key Findings

1. **DistilBERT** menunjukkan performa terbaik (97.40%) dengan kecepatan lebih tinggi dari BERT
2. **BERT** memiliki precision sempurna (1.00) untuk deteksi FAKE news
3. **LSTM** memberikan performa kompetitif (91.95%) dengan resource minimal
4. Semua model menunjukkan **balanced performance** antara FAKE dan REAL detection

---

## 📁 Struktur Proyek


Fake-Valid-News-Detection/
│
├── 📄 app.py                    # Main Streamlit application
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # Documentation
├── 📄 dataset.csv               # Dataset file
│
├── 📁 bert_model/               # Fine-tuned BERT model
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files
│
├── 📁 distilbert_model/         # Fine-tuned DistilBERT model
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files
│
├── 📄 lstm_model.keras          # Trained LSTM model
├── 📄 lstm_tokenizer.pkl        # LSTM tokenizer
├── 📄 model_config.pkl          # Model configuration
│
└── 📁 notebooks/                # Jupyter notebooks (training)
├── BERT_Training.ipynb
├── DistilBERT_Training.ipynb
└── LSTM_Training.ipynb

---

##  Panduan Menjalankan Sistem

### Prasyarat

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.8 | 3.10+ |
| **RAM** | 8 GB | 16 GB |
| **Storage** | 5 GB | 10 GB |
| **GPU** | - | NVIDIA (CUDA) |

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
bashDownloadCopy codestreamlit run app.py
6. Akses Website
Buka browser dan akses:
http://localhost:8501


🖥️ Screenshot Aplikasi
Halaman Utama
[Tambahkan screenshot halaman utama]
Hasil Deteksi - Real News
[Tambahkan screenshot hasil deteksi real]
Hasil Deteksi - Fake News
[Tambahkan screenshot hasil deteksi fake]
Model Comparison
[Tambahkan screenshot comparison]

 Troubleshooting
ProblemSolutionCUDA out of memoryGunakan CPU mode atau reduce batch sizeModel not foundJalankan python download_models.pyTokenizer errorPastikan versi transformers compatibleStreamlit errorUpdate streamlit: pip install --upgrade streamlit

 Dependencies
txtDownloadCopy codestreamlit>=1.0.0
torch>=1.9.0
transformers>=4.0.0
tensorflow>=2.8.0
pandas>=1.3.0
numpy>=1.21.0
plotly>=5.0.0
scikit-learn>=0.24.0

 Contributors

* [Nama Anda] - Initial work - GitHub


📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

 Acknowledgments

* Hugging Face for pre-trained models
* Kaggle for the dataset
* Streamlit for the web framework
