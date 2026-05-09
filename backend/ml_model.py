"""
=============================================================
  Module Machine Learning - Sistem Statistik Prediktif
  Menggunakan Linear Regression dari scikit-learn
=============================================================
  Modul ini menangani:
  - Memuat dan memproses dataset CSV
  - Melatih model Linear Regression
  - Melakukan prediksi GDP Growth
  - Mengevaluasi performa model (MAE, MSE, R²)
=============================================================
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class PredictiveModel:
    """
    Kelas utama untuk model prediktif GDP Growth.
    Menggunakan Linear Regression dengan fitur ekonomi makro.
    """

    def __init__(self, dataset_path=None):
        """
        Inisialisasi model prediktif.
        
        Args:
            dataset_path (str): Path ke file CSV dataset
        """
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.metrics = {}
        self.feature_names = []
        self.prediction_history = []
        self.dataset_path = dataset_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred_test = None

        # Muat dan latih model jika dataset tersedia
        if dataset_path and os.path.exists(dataset_path):
            self.load_and_train(dataset_path)

    def load_and_train(self, dataset_path):
        """
        Memuat dataset CSV dan melatih model secara otomatis.
        
        Args:
            dataset_path (str): Path ke file CSV dataset
            
        Returns:
            dict: Metrik evaluasi model
        """
        try:
            # ---- 1. Memuat Dataset ----
            self.df = pd.read_csv(dataset_path)
            self.dataset_path = dataset_path
            print(f"[INFO] Dataset dimuat: {self.df.shape[0]} baris, {self.df.shape[1]} kolom")

            # ---- 2. Memisahkan Fitur dan Target ----
            # Target: GDP_Growth_Persen
            # Fitur: semua kolom numerik kecuali target
            target_col = 'GDP_Growth_Persen'

            if target_col not in self.df.columns:
                raise ValueError(f"Kolom target '{target_col}' tidak ditemukan dalam dataset")

            # Pilih kolom fitur (semua kecuali target)
            self.feature_names = [col for col in self.df.columns if col != target_col]
            
            X = self.df[self.feature_names].values
            y = self.df[target_col].values

            # ---- 3. Split Data Training dan Testing (80:20) ----
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # ---- 4. Normalisasi Fitur dengan StandardScaler ----
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)

            # ---- 5. Melatih Model Linear Regression ----
            self.model = LinearRegression()
            self.model.fit(self.X_train, self.y_train)
            self.is_trained = True
            print("[INFO] Model berhasil dilatih!")

            # ---- 6. Evaluasi Model ----
            self.y_pred_test = self.model.predict(self.X_test)
            self._calculate_metrics()

            return self.metrics

        except Exception as e:
            print(f"[ERROR] Gagal melatih model: {str(e)}")
            raise e

    def _calculate_metrics(self):
        """
        Menghitung metrik evaluasi model:
        - MAE (Mean Absolute Error)
        - MSE (Mean Squared Error)
        - R² Score (Koefisien Determinasi)
        """
        if self.y_test is not None and self.y_pred_test is not None:
            self.metrics = {
                'mae': round(mean_absolute_error(self.y_test, self.y_pred_test), 4),
                'mse': round(mean_squared_error(self.y_test, self.y_pred_test), 4),
                'r2_score': round(r2_score(self.y_test, self.y_pred_test), 4),
                'train_size': len(self.y_train),
                'test_size': len(self.y_test),
                'total_data': len(self.y_train) + len(self.y_test)
            }

    def predict(self, input_data):
        """
        Melakukan prediksi GDP Growth berdasarkan input pengguna.
        
        Args:
            input_data (dict): Dictionary berisi nilai fitur
                Contoh: {
                    'Tahun': 2025,
                    'Kuartal': 1,
                    'Populasi_Juta': 280,
                    ...
                }
                
        Returns:
            dict: Hasil prediksi beserta detail
        """
        if not self.is_trained:
            raise Exception("Model belum dilatih. Silakan muat dataset terlebih dahulu.")

        try:
            # Susun input sesuai urutan fitur
            input_values = []
            for feature in self.feature_names:
                if feature in input_data:
                    input_values.append(float(input_data[feature]))
                else:
                    raise ValueError(f"Fitur '{feature}' tidak ditemukan dalam input")

            # Transformasi input dan prediksi
            input_array = np.array(input_values).reshape(1, -1)
            input_scaled = self.scaler.transform(input_array)
            prediction = self.model.predict(input_scaled)[0]

            # Simpan ke riwayat
            result = {
                'input': input_data,
                'prediction': round(float(prediction), 4),
                'unit': '% (Pertumbuhan GDP)',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model': 'Linear Regression'
            }
            self.prediction_history.append(result)

            return result

        except Exception as e:
            raise Exception(f"Gagal melakukan prediksi: {str(e)}")

    def get_metrics(self):
        """Mengembalikan metrik evaluasi model."""
        return self.metrics

    def get_history(self):
        """Mengembalikan riwayat prediksi."""
        return self.prediction_history

    def get_dataset_info(self):
        """
        Mengembalikan informasi dataset untuk ditampilkan di frontend.
        
        Returns:
            dict: Info dataset termasuk preview data
        """
        if self.df is None:
            return None

        return {
            'columns': self.df.columns.tolist(),
            'shape': {'rows': self.df.shape[0], 'cols': self.df.shape[1]},
            'data': self.df.to_dict(orient='records'),
            'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            'statistics': json.loads(self.df.describe().to_json()),
            'feature_names': self.feature_names,
            'target': 'GDP_Growth_Persen'
        }

    def get_model_info(self):
        """
        Mengembalikan informasi detail tentang model.
        
        Returns:
            dict: Detail model termasuk koefisien dan intercept
        """
        if not self.is_trained:
            return None

        # Koefisien model (bobot setiap fitur)
        coefficients = {}
        for name, coef in zip(self.feature_names, self.model.coef_):
            coefficients[name] = round(float(coef), 6)

        return {
            'model_type': 'Linear Regression',
            'library': 'scikit-learn',
            'features': self.feature_names,
            'target': 'GDP_Growth_Persen',
            'coefficients': coefficients,
            'intercept': round(float(self.model.intercept_), 6),
            'metrics': self.metrics,
            'scaler': 'StandardScaler',
            'test_split': '20%',
            'train_split': '80%',
            'description': 'Model Linear Regression untuk memprediksi pertumbuhan GDP berdasarkan indikator ekonomi makro Indonesia.',
            'actual_vs_predicted': {
                'actual': [round(float(v), 4) for v in self.y_test],
                'predicted': [round(float(v), 4) for v in self.y_pred_test]
            }
        }

    def get_trend_data(self):
        """
        Mengembalikan data tren untuk grafik di dashboard.
        
        Returns:
            dict: Data tren GDP dan variabel lainnya
        """
        if self.df is None:
            return None

        # Label sumbu X: Tahun-Q{Kuartal}
        labels = [f"{int(row['Tahun'])}-Q{int(row['Kuartal'])}" for _, row in self.df.iterrows()]

        return {
            'labels': labels,
            'gdp_growth': self.df['GDP_Growth_Persen'].tolist(),
            'inflasi': self.df['Inflasi_Persen'].tolist(),
            'pengangguran': self.df['Pengangguran_Persen'].tolist(),
            'investasi': self.df['Investasi_Triliun'].tolist(),
            'ekspor': self.df['Ekspor_Miliar_USD'].tolist(),
            'suku_bunga': self.df['Suku_Bunga_Persen'].tolist()
        }

    def clear_history(self):
        """Menghapus riwayat prediksi."""
        self.prediction_history = []
