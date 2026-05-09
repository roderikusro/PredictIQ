# 🧠 Projek Akhir BPS — Sistem Statistik Prediktif

Sistem Statistik Prediktif Berbasis Machine Learning untuk analisis dan prediksi data ekonomi Indonesia.

> **Project Akhir Magang** — Fokus pada statistik prediktif sederhana namun profesional.

---

## 📸 Fitur Utama

- **Dashboard Modern** — Ringkasan statistik, grafik tren GDP, dan indikator ekonomi
- **Prediksi GDP Growth** — Input parameter ekonomi dan dapatkan prediksi realtime
- **Evaluasi Model** — MAE, MSE, R² Score dengan visualisasi
- **Dataset Viewer** — Preview data, statistik deskriptif, dan pencarian
- **Upload CSV** — Upload dataset baru untuk melatih ulang model
- **Export Prediksi** — Download riwayat prediksi dalam format CSV
- **Dark Mode** — UI gelap modern dengan efek glassmorphism

---

## 🛠️ Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Frontend | HTML5, CSS3, JavaScript (ES6+) |
| Backend | Python Flask |
| Machine Learning | scikit-learn (Linear Regression) |
| Visualisasi | Chart.js |
| Icons | Lucide Icons |
| Font | Inter (Google Fonts) |

---

## 📁 Struktur Project

```
Projek Akhir/
├── backend/
│   ├── app.py              # Flask server & API endpoints
│   └── ml_model.py         # Modul Machine Learning
├── dataset/
│   └── ekonomi_data.csv    # Dataset dummy ekonomi Indonesia
├── static/
│   ├── css/
│   │   └── style.css       # Stylesheet utama
│   └── js/
│       └── app.js          # JavaScript aplikasi
├── templates/
│   └── index.html          # Template HTML (SPA)
├── uploads/                # Folder upload CSV (auto-created)
├── requirements.txt        # Dependencies Python
└── README.md               # Dokumentasi
```

---

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan Server

```bash
cd backend
python app.py
```

### 3. Buka Browser

Akses `http://localhost:5000`

---

## 📊 API Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/api/dashboard` | Data ringkasan dashboard |
| GET | `/api/dataset` | Data dataset lengkap |
| GET | `/api/model-info` | Informasi model ML |
| GET | `/api/trend` | Data tren untuk grafik |
| GET | `/api/history` | Riwayat prediksi |
| POST | `/api/predict` | Melakukan prediksi |
| POST | `/api/upload-csv` | Upload dataset baru |
| GET | `/api/export` | Export riwayat (CSV) |
| POST | `/api/clear-history` | Hapus riwayat |

---

## 🤖 Model Machine Learning

- **Algoritma**: Linear Regression (scikit-learn)
- **Target**: GDP_Growth_Persen (Pertumbuhan GDP per kuartal)
- **Fitur**: 9 variabel ekonomi makro (Tahun, Kuartal, Populasi, Inflasi, Suku Bunga, Pengangguran, Investasi, Ekspor, Konsumsi RT)
- **Preprocessing**: StandardScaler
- **Split Data**: 80% training, 20% testing
- **Evaluasi**: MAE, MSE, R² Score

---

## 👨‍💻 Dibuat Untuk

Project akhir magang mahasiswa — Sistem Statistik Prediktif berbasis Machine Learning.
