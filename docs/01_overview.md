# 📘 Technical Documentation — Sistem Statistik Prediktif

## 1. Gambaran Umum Sistem

**Nama Proyek:** Sistem Statistik Prediktif Berbasis Machine Learning  
**Tujuan:** Menganalisis dan memprediksi data ekonomi Indonesia menggunakan algoritma Machine Learning  
**Konteks:** Project Akhir Magang — Badan Pusat Statistik (BPS)

### 1.1 Deskripsi Singkat

Sistem ini adalah aplikasi web berbasis Flask yang menyediakan kemampuan prediksi data ekonomi menggunakan dua algoritma ML: **Linear Regression** dan **Random Forest Regression**. Aplikasi dibangun sebagai Single Page Application (SPA) dengan antarmuka modern dark-mode dan fitur interaktif seperti dashboard, EDA, prediksi realtime, perbandingan model, dan manajemen dataset.

### 1.2 Fitur Utama

| No | Fitur | Deskripsi |
|----|-------|-----------|
| 1 | Dashboard Interaktif | Ringkasan statistik, grafik tren, indikator ekonomi |
| 2 | Exploratory Data Analysis (EDA) | Korelasi, outlier, skewness, distribusi, scatter plot |
| 3 | Prediksi Single Model | Prediksi menggunakan LR atau RF secara individual |
| 4 | Prediksi Perbandingan | Prediksi menggunakan kedua model sekaligus + insight otomatis |
| 5 | Evaluasi Model | MAE, MSE, RMSE, R² Score dengan visualisasi |
| 6 | Dataset Viewer | Preview data, statistik deskriptif, sorting, pagination, pencarian |
| 7 | Konfigurasi Model | Pilih fitur, target, dan kolom waktu secara dinamis |
| 8 | Upload CSV | Upload dataset baru, otomatis melatih ulang model |
| 9 | Export Prediksi | Download riwayat prediksi dalam format CSV |
| 10 | Dark Mode UI | Antarmuka gelap modern dengan efek glassmorphism |
| 11 | Otomasi Infografis | Generate infografis dinamis (3 layout) dan ekspor ke PNG rasio 9:16 |

### 1.3 Tech Stack

| Komponen | Teknologi | Versi |
|----------|-----------|-------|
| Backend | Python Flask | 3.1.1 |
| CORS | Flask-CORS | 5.0.1 |
| Machine Learning | scikit-learn | 1.6.1 |
| Data Processing | Pandas | 2.2.3 |
| Numerical Computing | NumPy | 2.2.6 |
| Serialization | Joblib | 1.5.0 |
| Frontend | HTML5, CSS3, JavaScript ES6+ | — |
| Charting | Chart.js | CDN |
| Export Image | html2canvas | CDN |
| Icons | Lucide Icons | CDN |
| Font | Inter (Google Fonts) | CDN |
| Deployment | PythonAnywhere (WSGI) | — |

---

## 2. Struktur Proyek

```
Projek Akhir/
├── backend/
│   ├── app.py                  # Flask server & API endpoints (404 baris)
│   └── ml_model.py             # Module Machine Learning (704 baris)
├── dataset/
│   ├── ekonomi_data.csv        # Dataset default ekonomi Indonesia (41 baris data)
│   └── Data Wendy.csv          # Dataset alternatif — indikator makroekonomi (73 baris data)
├── static/
│   ├── css/
│   │   └── style.css           # Stylesheet utama (28KB)
│   └── js/
│       └── app.js              # JavaScript aplikasi SPA (1565 baris)
├── templates/
│   └── index.html              # Template HTML Single Page Application (53KB)
├── uploads/                    # Folder upload CSV (auto-created)
├── requirements.txt            # Dependencies Python
├── wsgi_pythonanywhere.py      # Konfigurasi WSGI untuk deployment
├── test_pred.py                # Script pengujian prediksi API
├── test_eda.py                 # Script pengujian endpoint EDA
├── test_comp.py                # Script pengujian perbandingan model
├── push.bat                    # Script otomatis push ke GitHub
├── .gitignore                  # Konfigurasi Git ignore
└── README.md                   # Dokumentasi ringkas
```
