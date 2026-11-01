# 🏥 Prediksi Length of Stay (LoS)

> Sistem prediksi berbasis Machine Learning untuk mengklasifikasikan lama rawat inap pasien di Rumah Sakit Umum Aulia, dilengkapi dengan **dashboard interaktif Streamlit** yang dapat digunakan langsung oleh tenaga medis dan manajemen rumah sakit.

---

## 📋 Daftar Isi

- [Latar Belakang](#latar-belakang)
- [Fitur Utama](#fitur-utama)
- [Struktur Proyek](#struktur-proyek)
- [Alur Pipeline](#alur-pipeline)
- [Instalasi](#instalasi)
- [Cara Menjalankan](#cara-menjalankan)
- [Dashboard](#dashboard)
- [Model & Evaluasi](#model--evaluasi)
- [Catatan Penting](#catatan-penting)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)

---

## 🎯 Latar Belakang

**Length of Stay (LoS)** atau lama rawat inap adalah salah satu indikator kritis dalam manajemen kapasitas rumah sakit. Prediksi LoS yang akurat memungkinkan:

- Perencanaan ketersediaan tempat tidur yang lebih baik
- Optimalisasi alokasi sumber daya medis
- Deteksi dini pasien berisiko dirawat lama (prioritas intervensi klinis)

Proyek ini membangun model klasifikasi multiclass berdasarkan **data rekam medis RSU Aulia**, mengkategorikan LoS ke dalam tiga kelas:

| Kategori | Definisi |
|----------|----------|
| 🟢 **Pendek** | LoS ≤ 2 hari |
| 🟡 **Sedang** | LoS 3–5 hari |
| 🔴 **Panjang** | LoS > 5 hari |

---

## ✨ Fitur Utama

- **Prediksi Individu** — Input data pasien secara manual dan dapatkan prediksi LoS beserta probabilitas dan penjelasan SHAP
- **Prediksi Massal** — Unggah file CSV/XLSX untuk memprediksi ratusan pasien sekaligus
- **Analisis Kapasitas & Tren** — Visualisasi beban per ruang perawatan berdasarkan data historis
- **Perbandingan Model** — Evaluasi mendalam XGBoost vs Random Forest dengan confusion matrix dan SHAP global
- **Explainability (SHAP)** — Setiap prediksi disertai penjelasan fitur mana yang paling berpengaruh

---

## 📁 Struktur Proyek

```
project_los_rsu_aulia/
│
├── 1_cleaning.py          # Pipeline pembersihan dan feature engineering data
├── 2_modeling.py          # Training model (XGBoost & Random Forest + SMOTENC)
├── 3_evaluation.py        # Evaluasi model, confusion matrix, dan SHAP analysis
│
├── pages/
│   └── app.py             # Aplikasi dashboard Streamlit (4 halaman)
│
├── data/
│   ├── rekam_medis.xlsx   # Data sumber (input mentah dari SIMRS)
│   ├── data_clean.csv     # Data setelah proses cleaning
│   ├── X_test.csv         # Fitur data test (untuk evaluasi)
│   ├── y_test.csv         # Label data test
│   ├── confusion_matrix.png
│   ├── shap_summary.png
│   ├── metrik.json        # Hasil metrik evaluasi (accuracy, F1, recall, dll.)
│   ├── cm.json            # Nilai confusion matrix dalam format JSON
│   └── shap_summary.json  # Nilai SHAP per fitur
│
├── dashboard/
│   ├── model.pkl          # Model terbaik (dipilih otomatis dari XGBoost/RF)
│   ├── model_xgb.pkl      # Model XGBoost tersimpan
│   ├── model_rf.pkl       # Model Random Forest tersimpan
│   ├── encoders.pkl       # Label encoders untuk semua fitur kategorikal
│   ├── fitur.pkl          # Daftar nama fitur yang digunakan
│   └── nama_model.pkl     # Nama model terbaik yang dipilih
│
└── requirements.txt       # Dependensi Python
```

---

## 🔄 Alur Pipeline

```
rekam_medis.xlsx
       │
       ▼
 1_cleaning.py        ──► data_clean.csv
       │
       ▼
 2_modeling.py        ──► model_xgb.pkl, model_rf.pkl,
       │                   model.pkl, encoders.pkl, fitur.pkl
       ▼
 3_evaluation.py      ──► confusion_matrix.png, shap_summary.png,
       │                   metrik.json, cm.json, shap_summary.json
       ▼
 pages/app.py         ──► Dashboard Streamlit (siap digunakan)
```

### Tahap 1 — Data Cleaning (`1_cleaning.py`)

- Standarisasi nama kolom dari SIMRS RSU Aulia
- Filter: hanya pasien **pulang atas persetujuan medis**
- Kalkulasi LoS dari selisih `tgl_masuk` dan `tgl_keluar`
- Penghapusan anomali LoS (> 19 hari) dan nilai negatif
- Deduplikasi kunjungan per `no_registrasi` (mengambil diagnosis utama)
- **Feature engineering**:
  - Parsing usia dari format teks `"31 Th 4 Bl 13 Hr"` ke angka desimal tahun
  - Ekstraksi `bab_diagnosis` dari huruf pertama kode ICD-10
  - Ekstraksi `hari_masuk` (Senin–Minggu) dari tanggal masuk
- Pelabelan target: `los_kategori` (0 = Pendek, 1 = Sedang, 2 = Panjang)

### Tahap 2 — Modeling (`2_modeling.py`)

- Split data: **80% train / 20% test** menggunakan `StratifiedGroupKFold` (grouped by `rm` untuk menghindari data leakage antar kunjungan pasien yang sama)
- **Label Encoding** pada fitur kategorikal (kompatibel dengan SMOTENC)
- **SMOTENC** untuk menangani ketidakseimbangan kelas minoritas ("Panjang")
- `GridSearchCV` untuk hyperparameter tuning pada dua model:
  - **XGBoost** (`XGBClassifier`, multi:softmax)
  - **Random Forest** (`RandomForestClassifier`, class_weight=`balanced`)
- Model terbaik dipilih otomatis berdasarkan **F1-macro** di test set

### Tahap 3 — Evaluasi (`3_evaluation.py`)

- Evaluasi lengkap: Accuracy, Precision, Recall, F1 (weighted & macro)
- **Confusion matrix** (absolut & ternormalisasi) untuk XGBoost dan Random Forest
- **SHAP TreeExplainer** untuk interpretabilitas model secara global

---

## 🚀 Instalasi

### Prasyarat

- Python 3.10 atau lebih baru
- (Disarankan) Virtual environment

### Langkah Instalasi

```bash
# 1. Clone repositori
git clone https://github.com/hyerocienpd/prediksi-los.git
cd prediksi-los

# 2. Buat dan aktifkan virtual environment (opsional tapi disarankan)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 3. Install semua dependensi
pip install -r requirements.txt

# 4. Install imbalanced-learn (diperlukan oleh 2_modeling.py)
pip install imbalanced-learn
```

---

## ▶️ Cara Menjalankan

Jalankan pipeline secara **berurutan** dari root direktori proyek:

### Langkah 1: Bersihkan Data

```bash
python 1_cleaning.py
```

Output: `data/data_clean.csv`

### Langkah 2: Latih Model

```bash
python 2_modeling.py
```

> ⏱️ Proses ini membutuhkan waktu beberapa menit karena GridSearchCV dengan cross-validation.

Output: model dan encoder tersimpan di folder `dashboard/`

### Langkah 3: Evaluasi Model

```bash
python 3_evaluation.py
```

Output: grafik dan metrik tersimpan di folder `data/`

### Langkah 4: Jalankan Dashboard

```bash
streamlit run pages/app.py
```

Buka browser dan akses: **http://localhost:8501**

---

## 🖥️ Dashboard

Dashboard terdiri dari **4 halaman** yang dapat diakses melalui sidebar:

### 1. 🔍 Prediksi Individu
Input data klinis satu pasien (usia, jenis kelamin, ruang perawatan, diagnosa ICD-10, dll.) dan dapatkan:
- Kategori prediksi LoS dengan badge warna
- Bar chart probabilitas tiap kelas
- Waterfall plot SHAP untuk menjelaskan alasan prediksi

### 2. 📂 Prediksi Massal
- Unggah file **CSV** atau **XLSX** yang berisi data banyak pasien
- Unduh **template file** yang sudah berformat benar
- Hasil prediksi bisa diunduh kembali dalam format CSV/XLSX

### 3. 📈 Kapasitas & Tren
- Pilih periode laporan (per bulan)
- Lihat total kunjungan, rata-rata LoS, dan jumlah pasien LoS panjang
- Visualisasi beban kategori per ruang perawatan (stacked bar)

### 4. 📊 Perbandingan Model
- Komparasi metrik XGBoost vs Random Forest (bar chart grouped)
- Recall kelas "Panjang" (kritis untuk kapasitas rumah sakit)
- Confusion matrix interaktif (Plotly)
- SHAP global importance per model

---

## 📊 Model & Evaluasi

### Fitur Input

| Fitur | Tipe | Keterangan |
|-------|------|------------|
| `usia` | Numerik | Usia dalam tahun (desimal) |
| `jenis_kelamin` | Kategorikal | Jenis kelamin pasien |
| `bab_diagnosis` | Kategorikal | Huruf pertama kode ICD-10 (Bab penyakit) |
| `ruang_perawatan` | Kategorikal | Nama ruang rawat inap |
| `jenis_perawatan` | Kategorikal | Jenis perawatan yang diterima |
| `cara_datang` | Kategorikal | Cara kedatangan (UGD, rujukan, dll.) |
| `hari_masuk` | Kategorikal | Hari masuk rawat inap (Senin–Minggu) |

### Target

`los_kategori` — kelas multiclass (0: Pendek, 1: Sedang, 2: Panjang)

### Metrik Evaluasi

Metrik utama yang digunakan adalah **F1-macro** karena mempertimbangkan performa pada seluruh kelas secara seimbang, termasuk kelas minoritas "Panjang".

> Metrik lengkap tersedia di halaman **Perbandingan Model** pada dashboard setelah pipeline dijalankan.

---

## ⚠️ Catatan Penting

1. **Keterbatasan Model**: Model ini dilatih pada data spesifik RSU Aulia. Performa dapat berbeda jika digunakan pada data rumah sakit lain dengan pola pasien yang berbeda.

2. **Kelas Minoritas**: Kelas "Panjang" (LoS > 5 hari) memiliki representasi terbatas dalam data. Meskipun SMOTENC digunakan, recall kelas ini masih terbatas. Gunakan prediksi sebagai **alat bantu pendukung keputusan klinis**, bukan sebagai pengganti penilaian tenaga medis.

3. **Neonatus**: Pasien dengan usia < 1 bulan (~3,9% dari data) memiliki keterwakilan kecil dalam data pelatihan. Prediksi untuk kelompok ini perlu diinterpretasikan dengan hati-hati.

4. **Pembaruan Model**: Jika data baru tersedia, jalankan ulang pipeline dari `1_cleaning.py` hingga `3_evaluation.py`, lalu klik tombol **🔄 Muat Ulang Model & Data** di sidebar dashboard.

5. **Data Privasi**: File `data/rekam_medis.xlsx` berisi data rekam medis. Pastikan tidak diupload ke repositori publik. Tambahkan ke `.gitignore` jika diperlukan:
   ```
   data/rekam_medis.xlsx
   data/data_clean.csv
   ```

---

## 🛠️ Teknologi yang Digunakan

| Teknologi | Versi | Kegunaan |
|-----------|-------|----------|
| [Streamlit](https://streamlit.io) | ≥ 1.35 | Framework dashboard interaktif |
| [Pandas](https://pandas.pydata.org) | ≥ 2.0 | Manipulasi dan analisis data |
| [NumPy](https://numpy.org) | ≥ 1.26 | Komputasi numerik |
| [scikit-learn](https://scikit-learn.org) | ≥ 1.4 | ML pipeline, GridSearchCV, evaluasi |
| [XGBoost](https://xgboost.readthedocs.io) | ≥ 2.0 | Model gradient boosting |
| [imbalanced-learn](https://imbalanced-learn.org) | — | SMOTENC untuk oversampling kelas minoritas |
| [SHAP](https://shap.readthedocs.io) | ≥ 0.45 | Model interpretability |
| [Plotly](https://plotly.com) | ≥ 5.20 | Visualisasi interaktif |
| [Joblib](https://joblib.readthedocs.io) | ≥ 1.3 | Serialisasi model |
| [openpyxl](https://openpyxl.readthedocs.io) | ≥ 3.1 | Baca/tulis file Excel |

---

## 📄 Lisensi

Proyek ini dikembangkan untuk keperluan penelitian akademik. Segala penggunaan untuk keperluan komersial atau klinis harus memperhatikan validasi lebih lanjut dan persetujuan pihak terkait.

---
