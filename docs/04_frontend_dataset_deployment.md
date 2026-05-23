# 📘 Technical Documentation — Frontend & Dataset

## 6. Frontend Architecture

### 6.1 Single Page Application (SPA)

Aplikasi menggunakan arsitektur SPA dengan navigasi berbasis JavaScript. Semua halaman berada dalam satu file `index.html` dengan class `.page` yang di-toggle melalui JavaScript.

#### Halaman (Pages)

| Page ID | Nama | Deskripsi |
|---------|------|-----------|
| `page-dashboard` | Dashboard | Ringkasan statistik, grafik tren, riwayat prediksi |
| `page-eda` | Analisis Data | EDA: korelasi, distribusi, scatter, outlier |
| `page-prediksi` | Prediksi | Form input prediksi, hasil, riwayat |
| `page-model` | Tentang Model | Detail model, koefisien, actual vs predicted |
| `page-comparison` | Perbandingan | Perbandingan kedua model side-by-side |
| `page-dataset` | Dataset | Viewer, konfigurasi model, upload CSV |
| `page-infografis` | Otomasi Infografis | Generator infografis otomatis dengan rasio statis 9:16 untuk diekspor sebagai PNG |

### 6.2 JavaScript Modules (`app.js` — 1565 baris)

#### Inisialisasi

```javascript
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();    // Sidebar navigation
  initMobileMenu();    // Responsive menu
  initPredictForm();   // Form prediksi
  initCompareForm();   // Form perbandingan
  initUpload();        // Upload CSV
  initExport();        // Export riwayat
  initClearHistory();  // Hapus riwayat
  initDatasetSearch(); // Pencarian dataset
  lucide.createIcons(); // Render ikon
  loadDashboard();     // Load data awal
});
```

#### Chart.js Instances

| Variable | Chart Type | Lokasi | Deskripsi |
|----------|-----------|--------|-----------|
| `chartGDP` | Line | Dashboard | Tren target variable |
| `chartIndicators` | Line | Dashboard | Tren indikator/fitur |
| `chartActualVsPred` | Bar | Model | Aktual vs Prediksi |
| `chartCoefficients` | Bar (horizontal) | Model | Koefisien/Feature Importance |
| `chartCompMetrics` | Radar | Comparison | Perbandingan metrik |
| `chartFeatureImp` | Doughnut | Comparison | Feature importance |
| `chartCompAVP` | Line | Comparison | Actual vs Predicted overlay |
| `chartEdaDist` | Bar | EDA | Distribusi variabel |
| `chartEdaScatter` | Scatter | EDA | Scatter plot |

#### Dynamic Form Generation

Form prediksi di-generate secara dinamis berdasarkan fitur model aktif:

```javascript
function renderDynamicForms() {
  // 1. Render time columns (jika ada) dengan styling khusus
  // 2. Render ML features sebagai input number
  // 3. Update label hasil berdasarkan nama target aktual
  // 4. Replace teks statis "GDP Growth" dengan nama target sebenarnya
}
```

#### Dataset Table Features

- **Sorting:** Click header kolom untuk sort ascending/descending
- **Search:** Filter real-time berdasarkan semua kolom
- **Pagination:** 10 baris per halaman dengan navigasi prev/next

### 6.3 Styling (`style.css` — 28KB)

#### Design System

| Token | Nilai | Penggunaan |
|-------|-------|-----------|
| `--bg-primary` | `#0a0e1a` | Background utama |
| `--bg-card` | `rgba(15,23,42,0.8)` | Background kartu |
| `--text-primary` | `#e2e8f0` | Teks utama |
| `--text-secondary` | `#94a3b8` | Teks sekunder |
| `--accent` | `#8b5cf6` | Warna aksen (violet) |
| `--success` | `#22c55e` | Status sukses |
| `--error` | `#ef4444` | Status error |

#### Efek Visual

- **Glassmorphism:** `backdrop-filter: blur()` pada kartu
- **Gradient:** Background gradient halus
- **Animations:** Fade-in untuk halaman, pulse untuk status dot
- **Responsive:** Sidebar collapsible untuk mobile

---

## 7. Dataset

### 7.1 Dataset Default (`ekonomi_data.csv`)

Dataset ekonomi makro Indonesia per kuartal (2015–2024).

| Kolom | Tipe | Satuan | Deskripsi |
|-------|------|--------|-----------|
| `Tahun` | int | — | Tahun data |
| `Kuartal` | int | 1–4 | Kuartal dalam tahun |
| `Populasi_Juta` | float | Juta jiwa | Jumlah penduduk |
| `Inflasi_Persen` | float | % | Tingkat inflasi |
| `Suku_Bunga_Persen` | float | % | Suku bunga BI |
| `Pengangguran_Persen` | float | % | Tingkat pengangguran |
| `Investasi_Triliun` | float | Triliun IDR | Total investasi |
| `Ekspor_Miliar_USD` | float | Miliar USD | Nilai ekspor |
| `Konsumsi_RT_Triliun` | float | Triliun IDR | Konsumsi rumah tangga |
| `GDP_Growth_Persen` | float | % | **TARGET** — Pertumbuhan GDP |

- **Jumlah baris:** 40 (Q1 2015 — Q4 2024)
- **Delimiter:** Koma (`,`)
- **Format desimal:** Titik (US standard)

### 7.2 Dataset Alternatif (`Data Wendy.csv`)

Dataset indikator ekonomi dan pasar modal Indonesia per bulan (2018–2023).

| Kolom | Tipe | Satuan | Deskripsi |
|-------|------|--------|-----------|
| `Tahun` | int | — | Tahun data |
| `Bulan` | int | 1–12 | Bulan dalam tahun |
| `Fed Rate` | float | % | Federal Reserve interest rate |
| `Inflasi` | string | % | Tingkat inflasi (format: "3.25 %") |
| `Uang Beredar` | float | Miliar IDR | Jumlah uang beredar (M2) |
| `Indeks Harga Konsumen (IHK)` | float | — | Consumer Price Index |
| `Suku Bunga` | float | % | Suku bunga BI |
| `Indeks Dow Jones` | float | — | Perubahan Dow Jones (desimal) |
| `Indeks Saham Gabungan (IHSG)` | string | — | Nilai IHSG (format: "6.605,63") |

- **Jumlah baris:** 72 (Jan 2018 — Des 2023)
- **Delimiter:** Semicolon (`;`)
- **Format desimal:** Koma (format Indonesia)
- **Catatan:** Memerlukan data cleaning otomatis (% symbol, format angka Indonesia)

---

## 8. Deployment

### 8.1 Lokal (Development)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Jalankan server
cd backend
python app.py

# 3. Akses di browser
# http://localhost:5000
```

### 8.2 PythonAnywhere (Production)

File `wsgi_pythonanywhere.py` menyediakan konfigurasi WSGI:

```python
project_path = '/home/yourusername/Projek-Akhir'
backend_path = os.path.join(project_path, 'backend')

# Tambahkan ke sys.path
sys.path.insert(0, project_path)
sys.path.insert(0, backend_path)

# Set working directory
os.chdir(backend_path)

from app import app as application
```

**Langkah deployment:**
1. Upload project ke PythonAnywhere via GitHub
2. Buat virtual environment dan install requirements
3. Set WSGI configuration file ke `wsgi_pythonanywhere.py`
4. Ubah `yourusername` ke username PythonAnywhere yang sebenarnya
5. Reload web app

### 8.3 Push ke GitHub

File `push.bat` menyediakan script otomatis:

```bat
git add .
git commit -m "update"
git push origin main
```

---

## 9. Testing

### 9.1 Test Scripts

| File | Fungsi |
|------|--------|
| `test_pred.py` | Pengujian endpoint prediksi (LR dan RF) |
| `test_eda.py` | Pengujian endpoint EDA |
| `test_comp.py` | Pengujian endpoint perbandingan model |

### 9.2 Contoh Test Prediksi

```python
import requests

url = "http://localhost:5000/api/predict"
data_lr = {
    "Tahun": 2025, "Kuartal": 1, "Populasi_Juta": 280.0,
    "Inflasi_Persen": 3.0, "Suku_Bunga_Persen": 5.75,
    "Pengangguran_Persen": 5.0, "Investasi_Triliun": 275.0,
    "Ekspor_Miliar_USD": 58.0, "Konsumsi_RT_Triliun": 1950.0,
    "model_type": "linear_regression"
}
data_rf = data_lr.copy()
data_rf["model_type"] = "random_forest"

print("LR:", requests.post(url, json=data_lr).json())
print("RF:", requests.post(url, json=data_rf).json())
```

---

## 10. Limitasi & Catatan Teknis

| No | Item | Deskripsi |
|----|------|-----------|
| 1 | **In-Memory State** | Prediction history hilang saat server restart (tidak ada database) |
| 2 | **Single Instance** | Model di-share antar semua request (tidak thread-safe untuk re-training) |
| 3 | **Dataset Size** | Dataset kecil (40–72 baris), bisa overfitting |
| 4 | **No Authentication** | Tidak ada sistem login/autentikasi |
| 5 | **No Input Sanitization** | Input numerik tidak divalidasi range-nya |
| 6 | **Static Hyperparameters** | RF hyperparameters hardcoded, tidak ada tuning otomatis |
| 7 | **Scaler per Model** | Setiap model punya scaler terpisah (memastikan independensi) |
