# 📘 Technical Documentation — Machine Learning Module

## 5. Machine Learning Module (`ml_model.py`)

### 5.1 Class `PredictiveModel`

Class utama yang meng-encapsulate seluruh logika machine learning. Mengelola dua model secara bersamaan: **Linear Regression** dan **Random Forest Regression**.

#### Constructor

```python
def __init__(self, dataset_path=None)
```

**Atribut yang diinisialisasi:**

| Atribut | Tipe | Deskripsi |
|---------|------|-----------|
| `lr_model` | `LinearRegression` | Instance model Linear Regression |
| `rf_model` | `RandomForestRegressor` | Instance model Random Forest |
| `lr_scaler` / `rf_scaler` | `StandardScaler` | Scaler terpisah untuk masing-masing model |
| `lr_metrics` / `rf_metrics` | `dict` | Metrik evaluasi masing-masing model |
| `is_trained` | `bool` | Status apakah model sudah dilatih |
| `feature_names` | `list` | Nama kolom fitur yang digunakan |
| `target_name` | `str` | Nama kolom target |
| `prediction_history` | `list` | Riwayat prediksi (in-memory) |
| `df` | `DataFrame` | Dataset yang dimuat |
| `time_cols` | `dict` | Mapping kolom waktu (year, quarter, month, day) |

### 5.2 Pipeline Training (`load_and_train`)

```python
def load_and_train(self, dataset_path=None, target_col=None, feature_cols=None, time_cols=None)
```

#### Tahapan Pipeline:

```
1. LOAD DATASET
   └─ pd.read_csv(path, sep=None, engine='python')
      └─ Auto-detect delimiter (koma, semicolon, tab)

2. DATA CLEANING
   ├─ Deteksi kolom objek/string
   ├─ Pembersihan format angka:
   │   ├─ Hapus karakter non-numerik (Rp, $, %, spasi)
   │   ├─ Konversi format Indonesia (1.000,50 → 1000.50)
   │   ├─ Konversi format US (1,000.50 → 1000.50)
   │   └─ Handle edge cases (ribuan vs desimal)
   ├─ pd.to_numeric(errors='coerce')
   └─ Konversi kolom jika >50% berhasil jadi numerik

3. IMPUTASI MISSING VALUES
   └─ NaN diisi dengan median kolom

4. FEATURE/TARGET SPLIT
   ├─ Target: kolom yang dipilih user ATAU kolom numerik terakhir
   └─ Features: semua kolom numerik kecuali target

5. TRAIN-TEST SPLIT
   └─ 80% training, 20% testing (random_state=42)

6. TRAINING LINEAR REGRESSION
   ├─ StandardScaler.fit_transform(X_train)
   ├─ LinearRegression().fit(X_train_scaled, y_train)
   └─ Hitung metrik evaluasi

7. TRAINING RANDOM FOREST
   ├─ StandardScaler.fit_transform(X_train)
   ├─ RandomForestRegressor(
   │     n_estimators=100,
   │     max_depth=8,
   │     min_samples_split=3,
   │     min_samples_leaf=2,
   │     random_state=42,
   │     n_jobs=-1
   │   ).fit(X_train_scaled, y_train)
   └─ Hitung metrik evaluasi
```

### 5.3 Algoritma yang Digunakan

#### Linear Regression

- **Library:** `sklearn.linear_model.LinearRegression`
- **Prinsip:** Mencari hubungan linear `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ`
- **Output:** Koefisien (bobot) untuk setiap fitur dan intercept
- **Kelebihan:** Mudah diinterpretasi, cepat, cocok untuk hubungan linear
- **Preprocessing:** StandardScaler (normalisasi z-score)

#### Random Forest Regression

- **Library:** `sklearn.ensemble.RandomForestRegressor`
- **Prinsip:** Ensemble dari 100 decision trees, prediksi = rata-rata output semua tree
- **Hyperparameters:**

| Parameter | Nilai | Penjelasan |
|-----------|-------|------------|
| `n_estimators` | 100 | Jumlah decision trees |
| `max_depth` | 8 | Kedalaman maksimum setiap tree |
| `min_samples_split` | 3 | Minimum sampel untuk split node |
| `min_samples_leaf` | 2 | Minimum sampel di leaf node |
| `random_state` | 42 | Seed untuk reprodusibilitas |
| `n_jobs` | -1 | Gunakan semua CPU cores |

- **Output:** Feature importance (tingkat kepentingan fitur)
- **Kelebihan:** Menangkap non-linearitas, robust terhadap outlier
- **Preprocessing:** StandardScaler (normalisasi z-score)

### 5.4 Metrik Evaluasi

```python
def _calc_metrics(self, y_true, y_pred):
```

| Metrik | Formula | Interpretasi |
|--------|---------|-------------|
| **MAE** | `mean(\|y_true - y_pred\|)` | Rata-rata error absolut. Semakin kecil semakin baik. |
| **MSE** | `mean((y_true - y_pred)²)` | Rata-rata kuadrat error. Sensitif terhadap outlier. |
| **RMSE** | `√MSE` | Akar dari MSE, dalam satuan yang sama dengan target. |
| **R² Score** | `1 - SS_res/SS_tot` | Proporsi variansi yang dijelaskan model. 1.0 = sempurna. |

### 5.5 Fungsi Prediksi

#### Single Model Prediction

```python
def predict(self, input_data, model_type='linear_regression')
```

Alur: Input dict → Extract values sesuai `feature_names` → Reshape ke array 2D → Scale dengan scaler model → Predict → Simpan ke history → Return result

#### Comparison Prediction

```python
def predict_comparison(self, input_data)
```

Menjalankan prediksi dengan **kedua model** sekaligus dan menghasilkan:
- Hasil prediksi LR dan RF
- Model terbaik (berdasarkan R² Score)
- Selisih prediksi
- Insight otomatis (4 kategori insight)

### 5.6 Insight Generator

```python
def _generate_insight(self, lr_pred, rf_pred, best_model, diff, lr_r2, rf_r2, lr_mae, rf_mae)
```

Menghasilkan insight otomatis berdasarkan:

1. **Best Model** — Model mana yang lebih baik berdasarkan R² Score
2. **Konsistensi Prediksi:**
   - Selisih < 0.5%: "Sangat mirip, konsistensi baik"
   - Selisih 0.5–1.5%: "Perbedaan moderat"
   - Selisih > 1.5%: "Perbedaan signifikan"
3. **Error Comparison** — Perbandingan MAE kedua model
4. **Interpretasi Ekonomi:**
   - Rata-rata > 5%: "Pertumbuhan kuat"
   - 3–5%: "Pertumbuhan moderat"
   - 0–3%: "Pertumbuhan lambat"
   - < 0%: "Potensi kontraksi"

### 5.7 Exploratory Data Analysis (`get_eda`)

```python
def get_eda(self)
```

Menghasilkan data EDA lengkap:

| Komponen | Metode | Output |
|----------|--------|--------|
| Descriptive Statistics | `df.describe()` | Mean, std, min, max, quartiles |
| Correlation Matrix | `df.corr()` | Matriks korelasi Pearson |
| Missing Values | `df.isnull().sum()` | Jumlah NaN per kolom |
| Outlier Detection | IQR Method (`Q1 - 1.5*IQR`, `Q3 + 1.5*IQR`) | Jumlah outlier per kolom |
| Skewness | `df.skew()` | Kemiringan distribusi |
| Auto-Insights | Rule-based analysis | Multikolinearitas, korelasi terkuat, outlier info |

### 5.8 Data Cleaning — Format Angka

Sistem memiliki pembersihan format angka yang cerdas untuk menangani berbagai format:

```
Input Format          → Output
"1.000,50"           → 1000.50   (Format Indonesia)
"1,000.50"           → 1000.50   (Format US)
"3.25 %"             → 3.25      (Hapus simbol persen)
"5351684,67"         → 5351684.67
"Rp 1.000.000"       → 1000000
"8.826.531"          → 8826531   (Ribuan dengan titik)
```

Logika utama (`clean_number_string`):
1. Hapus karakter non-numerik kecuali `.`, `,`, `-`
2. Jika ada titik DAN koma → deteksi format Indonesia vs US berdasarkan posisi terakhir
3. Jika hanya koma → tentukan apakah desimal atau ribuan berdasarkan panjang digit
4. Jika hanya titik → tentukan apakah desimal atau ribuan berdasarkan panjang digit
5. Kolom dikonversi ke numerik jika >50% nilai berhasil dikonversi
