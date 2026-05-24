# 📘 TECHNICAL DOCUMENTATION
# Sistem Statistik Prediktif Berbasis Machine Learning
### Project Akhir Magang — Badan Pusat Statistik (BPS)

---

> **Versi:** 2.0  
> **Tanggal:** 25 Mei 2026  
> **Tech Stack:** Python Flask + scikit-learn + xgboost + Chart.js

---

## 📑 Daftar Isi

Dokumentasi ini terbagi dalam 5 bagian:

| No | File | Konten |
|----|------|--------|
| 1 | [01_overview.md](./01_overview.md) | Gambaran umum, fitur, tech stack, struktur proyek |
| 2 | [02_architecture_api.md](./02_architecture_api.md) | Arsitektur sistem, data flow, API reference lengkap |
| 3 | [03_ml_module.md](./03_ml_module.md) | Machine Learning module, algoritma, metrik, data cleaning |
| 4 | [04_frontend_dataset_deployment.md](./04_frontend_dataset_deployment.md) | Frontend, dataset, deployment, testing, limitasi |
| 5 | [05_technical_statistics_ml.md](./05_technical_statistics_ml.md) | Statistika mendalam: formula matematika, blended forecast, forecast reliability, target guardrails, EDA |

---

## 🔍 Ringkasan Cepat

**Apa ini?**  
Aplikasi web untuk memprediksi data ekonomi Indonesia menggunakan Linear Regression, Random Forest, dan XGBoost, dilengkapi Exploratory Data Analysis (EDA), auto-forecast multi-step, dan infografis otomatis.

**Bagaimana cara kerjanya?**
1. Dataset CSV/Excel dimuat dan dibersihkan secara otomatis (format angka Indonesia/US, imputasi median)
2. Tiga model ML dilatih secara paralel (LR + RF + XGBoost) dengan StandardScaler terpisah
3. User memasukkan parameter ekonomi via form dinamis atau menggunakan auto-forecast
4. Sistem memberikan prediksi + insight otomatis + forecast reliability assessment
5. Blended Forecast menggabungkan prediksi model terbaik dengan baseline historis (bobot adaptif berdasarkan R² dan Relative MAE)
6. Semua divisualisasikan dengan Chart.js (scatter, line, bar, radar, dan sparkline)

**Komponen utama:**
- `backend/app.py` — Flask API server (16 endpoints)
- `backend/ml_model.py` — Engine ML (class `PredictiveModel`, 1358 baris)
- `templates/index.html` — SPA frontend
- `static/js/app.js` — Logika frontend
- `static/css/style.css` — Dark mode UI dengan glassmorphism

**API Endpoints:**

| # | Method | Endpoint | Deskripsi |
|---|--------|----------|-----------|
| 1 | GET | `/` | Halaman utama (SPA) |
| 2 | GET | `/api/dashboard` | Data ringkasan dashboard |
| 3 | GET | `/api/dataset` | Data dataset lengkap |
| 4 | GET | `/api/model-info` | Informasi detail model ML |
| 5 | GET | `/api/model-comparison` | Perbandingan performa kedua model |
| 6 | GET | `/api/eda` | Exploratory Data Analysis (korelasi, outlier, skewness, insight) |
| 7 | GET | `/api/export-eda` | Export statistik deskriptif EDA (CSV/Excel) |
| 8 | GET | `/api/trend` | Data tren + auto-forecast untuk grafik |
| 9 | POST | `/api/predict` | Prediksi single model (LR, RF, atau XGBoost) |
| 10 | POST | `/api/predict-compare` | Prediksi perbandingan ketiga model + insight |
| 11 | GET | `/api/history` | Riwayat semua prediksi |
| 12 | POST | `/api/upload-csv` | Upload dataset CSV/Excel baru |
| 13 | POST | `/api/configure-model` | Konfigurasi ulang target, fitur, dan kolom waktu |
| 14 | GET | `/api/export` | Export riwayat prediksi (CSV/Excel) |
| 15 | GET | `/api/infographic-data` | Data lengkap untuk infografis otomatis |
| 16 | POST | `/api/clear-history` | Hapus riwayat prediksi |

**Quick Start:**
```bash
pip install -r requirements.txt
cd backend && python app.py
# → http://localhost:5000
```
