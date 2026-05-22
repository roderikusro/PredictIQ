# 📘 Technical Documentation — Arsitektur Sistem

## 3. Arsitektur Sistem

### 3.1 Diagram Arsitektur

```
┌──────────────────────────────────────────────────────┐
│                   CLIENT (Browser)                    │
│  ┌──────────┐  ┌──────────┐  ┌─────────────────────┐ │
│  │ index.html│  │ style.css│  │      app.js         │ │
│  │  (SPA)   │  │ (28KB)   │  │ (Chart.js, Lucide)  │ │
│  └────┬─────┘  └──────────┘  └──────────┬──────────┘ │
│       │              fetch() / AJAX      │            │
└───────┼──────────────────────────────────┼────────────┘
        │              HTTP REST API       │
        ▼                                  ▼
┌──────────────────────────────────────────────────────┐
│                 BACKEND (Flask Server)                 │
│  ┌─────────────────────────────────────────────────┐  │
│  │                  app.py                          │  │
│  │  • Route handler (/api/*)                       │  │
│  │  • File upload management                       │  │
│  │  • CSV export generation                        │  │
│  │  • Request validation                           │  │
│  └─────────────────┬───────────────────────────────┘  │
│                    │ import                            │
│  ┌─────────────────▼───────────────────────────────┐  │
│  │               ml_model.py                        │  │
│  │  • PredictiveModel class                        │  │
│  │  • Data loading & preprocessing                 │  │
│  │  • Linear Regression training                   │  │
│  │  • Random Forest training                       │  │
│  │  • Prediction & comparison                      │  │
│  │  • EDA (Exploratory Data Analysis)              │  │
│  │  • Metrics calculation                          │  │
│  └─────────────────┬───────────────────────────────┘  │
│                    │                                   │
│  ┌─────────────────▼───────────────────────────────┐  │
│  │            scikit-learn / pandas / numpy          │  │
│  └─────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────┐    ┌──────────────────┐
│  dataset/*.csv   │    │   uploads/*.csv   │
│ (Default Dataset)│    │  (User Uploads)   │
└──────────────────┘    └──────────────────┘
```

### 3.2 Pola Arsitektur

- **Pola:** Monolithic SPA + REST API
- **Frontend:** Single Page Application — satu file `index.html` dengan navigasi JavaScript
- **Backend:** Flask sebagai API server, melayani halaman HTML dan endpoint REST
- **ML Engine:** Class `PredictiveModel` sebagai encapsulation semua logika Machine Learning
- **State Management:** In-memory (prediction history, model state disimpan di RAM)

### 3.3 Alur Data (Data Flow)

```
[CSV File] → [pandas.read_csv] → [Data Cleaning] → [Feature/Target Split]
     → [train_test_split 80:20] → [StandardScaler] → [Model Training]
     → [LR Model + RF Model] → [Metrics Calculation]
     → [API Response → JSON] → [Chart.js Visualization]
```

### 3.4 Request-Response Flow

```
Browser                    Flask (app.py)              PredictiveModel (ml_model.py)
   │                            │                              │
   │── GET /api/dashboard ─────>│                              │
   │                            │── get_metrics() ────────────>│
   │                            │── get_dataset_info() ───────>│
   │                            │── get_history() ────────────>│
   │                            │<── return dict ──────────────│
   │<── JSON Response ─────────│                              │
   │                            │                              │
   │── POST /api/predict ──────>│                              │
   │   {features + model_type}  │── predict(input, type) ─────>│
   │                            │<── return prediction ────────│
   │<── JSON {prediction} ─────│                              │
```

---

## 4. Backend — Detail Komponen

### 4.1 Flask Application (`app.py`)

File ini merupakan entry point aplikasi. Bertanggung jawab atas:
- Konfigurasi Flask (template folder, static folder, CORS)
- Mendefinisikan semua API endpoint
- Inisialisasi model ML saat server start
- Handling file upload dan CSV export

#### Konfigurasi Penting

```python
# Path resolution relatif terhadap lokasi app.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
DEFAULT_DATASET = os.path.join(DATASET_DIR, 'ekonomi_data.csv')
```

#### Inisialisasi Model

Saat server dimulai, model langsung dimuat dan dilatih dengan dataset default:

```python
model = PredictiveModel(dataset_path=DEFAULT_DATASET)
```

### 4.2 API Endpoint Reference

| Method | Endpoint | Deskripsi | Request Body | Response |
|--------|----------|-----------|-------------|----------|
| `GET` | `/` | Render halaman SPA | — | HTML |
| `GET` | `/api/dashboard` | Data ringkasan dashboard | — | `{metrics, dataset_rows, features, ...}` |
| `GET` | `/api/dataset` | Data dataset lengkap | — | `{columns, data, statistics, ...}` |
| `GET` | `/api/model-info` | Info model (`?type=linear_regression\|random_forest`) | — | `{model_type, metrics, coefficients, ...}` |
| `GET` | `/api/model-comparison` | Perbandingan performa kedua model | — | `{lr, rf, best_model, summary}` |
| `GET` | `/api/eda` | Data Exploratory Data Analysis | — | `{correlation, outliers, skewness, ...}` |
| `GET` | `/api/trend` | Data tren untuk grafik | — | `{labels, target_data, indicators}` |
| `POST` | `/api/predict` | Prediksi single model | `{feature_values, model_type}` | `{prediction, model, timestamp}` |
| `POST` | `/api/predict-compare` | Prediksi kedua model | `{feature_values}` | `{lr_pred, rf_pred, best, insight}` |
| `GET` | `/api/history` | Riwayat prediksi | — | `{data: [...], total: N}` |
| `POST` | `/api/upload-csv` | Upload dataset CSV baru | `multipart/form-data` | `{metrics}` |
| `POST` | `/api/configure-model` | Konfigurasi ulang model | `{target, features, time_cols}` | `{metrics}` |
| `GET` | `/api/export` | Export riwayat ke CSV | — | CSV file download |
| `POST` | `/api/clear-history` | Hapus riwayat prediksi | — | `{message}` |

### 4.3 Contoh Request & Response

#### POST `/api/predict`

**Request:**
```json
{
  "Tahun": 2025,
  "Kuartal": 1,
  "Populasi_Juta": 280.0,
  "Inflasi_Persen": 3.0,
  "Suku_Bunga_Persen": 5.75,
  "Pengangguran_Persen": 5.0,
  "Investasi_Triliun": 275.0,
  "Ekspor_Miliar_USD": 58.0,
  "Konsumsi_RT_Triliun": 1950.0,
  "model_type": "linear_regression"
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "data": {
    "input": { "Tahun": 2025, "Kuartal": 1, ... },
    "prediction": 5.1234,
    "unit": "% (Pertumbuhan GDP)",
    "timestamp": "2026-05-22 09:00:00",
    "model": "Linear Regression",
    "model_type": "linear_regression"
  }
}
```

#### POST `/api/predict-compare`

**Response (200 OK):**
```json
{
  "status": "success",
  "data": {
    "linear_regression": { "prediction": 5.12, "metrics": {...} },
    "random_forest": { "prediction": 5.08, "metrics": {...} },
    "best_model": "Random Forest",
    "best_prediction": 5.08,
    "difference": 0.04,
    "insight": ["Random Forest menunjukkan performa lebih baik...", ...],
    "timestamp": "2026-05-22 09:00:00"
  }
}
```
