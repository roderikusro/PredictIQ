# 📘 TECHNICAL DOCUMENTATION
# Sistem Statistik Prediktif Berbasis Machine Learning
### Project Akhir Magang — Badan Pusat Statistik (BPS)

---

> **Versi:** 1.0  
> **Tanggal:** 22 Mei 2026  
> **Tech Stack:** Python Flask + scikit-learn + Chart.js

---

## 📑 Daftar Isi

Dokumentasi ini terbagi dalam 4 bagian:

| No | File | Konten |
|----|------|--------|
| 1 | [01_overview.md](./01_overview.md) | Gambaran umum, fitur, tech stack, struktur proyek |
| 2 | [02_architecture_api.md](./02_architecture_api.md) | Arsitektur sistem, data flow, API reference lengkap |
| 3 | [03_ml_module.md](./03_ml_module.md) | Machine Learning module, algoritma, metrik, data cleaning |
| 4 | [04_frontend_dataset_deployment.md](./04_frontend_dataset_deployment.md) | Frontend, dataset, deployment, testing, limitasi |

---

## 🔍 Ringkasan Cepat

**Apa ini?**  
Aplikasi web untuk memprediksi data ekonomi Indonesia menggunakan Linear Regression dan Random Forest.

**Bagaimana cara kerjanya?**
1. Dataset CSV dimuat dan dibersihkan secara otomatis
2. Dua model ML dilatih secara paralel (LR + RF)
3. User memasukkan parameter ekonomi via form dinamis
4. Sistem memberikan prediksi + insight otomatis
5. Semua divisualisasikan dengan Chart.js

**Komponen utama:**
- `backend/app.py` — Flask API server (14 endpoints)
- `backend/ml_model.py` — Engine ML (class `PredictiveModel`)
- `templates/index.html` — SPA frontend
- `static/js/app.js` — Logika frontend (1565 baris)
- `static/css/style.css` — Dark mode UI dengan glassmorphism

**Quick Start:**
```bash
pip install -r requirements.txt
cd backend && python app.py
# → http://localhost:5000
```
