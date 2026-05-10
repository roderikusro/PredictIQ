"""
=============================================================
  Module Machine Learning - Sistem Statistik Prediktif
  Menggunakan Linear Regression & Random Forest dari scikit-learn
=============================================================
  Modul ini menangani:
  - Memuat dan memproses dataset CSV
  - Melatih model Linear Regression & Random Forest
  - Melakukan prediksi GDP Growth (single & comparison)
  - Mengevaluasi performa model (MAE, RMSE, R²)
  - Membandingkan performa antar model
=============================================================
"""

import pandas as pd
import numpy as np
import os
import json
import math
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class PredictiveModel:
    """
    Kelas utama untuk model prediktif GDP Growth.
    Mendukung Linear Regression dan Random Forest Regression.
    """

    def __init__(self, dataset_path=None):
        """
        Inisialisasi model prediktif.
        
        Args:
            dataset_path (str): Path ke file CSV dataset
        """
        # Linear Regression
        self.lr_model = None
        self.lr_scaler = StandardScaler()
        self.lr_metrics = {}
        self.lr_pred_test = None

        # Random Forest
        self.rf_model = None
        self.rf_scaler = StandardScaler()
        self.rf_metrics = {}
        self.rf_pred_test = None

        # Shared state
        self.is_trained = False
        self.feature_names = []
        self.target_name = None
        self.prediction_history = []
        self.dataset_path = dataset_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.time_cols = {}

        # Muat dan latih model jika dataset tersedia
        if dataset_path and os.path.exists(dataset_path):
            self.load_and_train(dataset_path)

    def load_and_train(self, dataset_path=None, target_col=None, feature_cols=None, time_cols=None):
        """
        Memuat dataset CSV (opsional jika sudah dimuat) dan melatih kedua model secara otomatis.
        
        Args:
            dataset_path (str, optional): Path ke file CSV dataset
            target_col (str, optional): Nama kolom target
            feature_cols (list, optional): List nama kolom fitur
            
        Returns:
            dict: Metrik evaluasi kedua model
        """
        try:
            # ---- 1. Memuat Dataset ----
            if dataset_path:
                self.df = pd.read_csv(dataset_path, sep=None, engine='python')
                self.dataset_path = dataset_path
                
                # Pembersihan cerdas: Coba paksa konversi kolom objek ke numerik dengan mempertahankan desimal
                def clean_number_string(val):
                    if pd.isna(val):
                        return val
                    val = str(val).strip()
                    # Menghapus spasi atau karakter mata uang seperti Rp, $, dll jika ada di awal/akhir
                    val = ''.join(c for c in val if c.isdigit() or c in ['.', ',', '-'])
                    
                    if '.' in val and ',' in val:
                        # Format 1.000,50 (Indonesia) atau 1,000.50 (US)
                        if val.rfind(',') > val.rfind('.'):
                            # Indonesia: titik sebagai ribuan, koma sebagai desimal
                            val = val.replace('.', '').replace(',', '.')
                        else:
                            # US: koma sebagai ribuan, titik sebagai desimal
                            val = val.replace(',', '')
                    elif ',' in val:
                        if val.count(',') == 1:
                            parts = val.split(',')
                            if len(parts[1]) == 3 and len(parts[0]) <= 3 and val.startswith('0') == False:
                                # Bisa ribuan "1,000" atau desimal "1,000". Kita biarkan jadi desimal saja 
                                # untuk amannya karena ini konteks Indonesia.
                                val = val.replace(',', '.')
                            else:
                                val = val.replace(',', '.')
                        else:
                            # > 1 koma, pasti ribuan "1,000,000"
                            val = val.replace(',', '')
                            
                    return val

                for col in self.df.columns:
                    if self.df[col].dtype == 'object':
                        # Terapkan pembersihan format angka
                        cleaned = self.df[col].apply(clean_number_string)
                        converted = pd.to_numeric(cleaned, errors='coerce')
                        
                        # Jika sebagian besar (>50%) berhasil dikonversi menjadi angka, anggap ini kolom numerik
                        if converted.notna().sum() > len(self.df) * 0.5:
                            self.df[col] = converted

                print(f"[INFO] Dataset dimuat: {self.df.shape[0]} baris, {self.df.shape[1]} kolom")
            elif self.df is None:
                raise ValueError("Dataset belum dimuat dan path tidak diberikan.")

            # Filter kolom numerik
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError("Tidak ditemukan kolom numerik dalam dataset untuk dilatih.")

            # Imputasi nilai kosong (NaN) dengan median untuk mencegah error pada Scaler
            for col in numeric_cols:
                if self.df[col].isna().any():
                    self.df[col].fillna(self.df[col].median(), inplace=True)

            # ---- 2. Memisahkan Fitur dan Target ----
            if time_cols is not None:
                self.time_cols = time_cols

            # Gunakan kolom terakhir sebagai target default jika tidak dispesifikasikan
            if target_col and target_col in numeric_cols:
                self.target_name = target_col
            else:
                self.target_name = numeric_cols[-1]

            if feature_cols:
                self.feature_names = [col for col in feature_cols if col in numeric_cols and col != self.target_name]
            else:
                self.feature_names = [col for col in numeric_cols if col != self.target_name]
            
            if not self.feature_names:
                raise ValueError("Minimal 1 fitur numerik harus dipilih untuk melatih model.")

            X = self.df[self.feature_names].values
            y = self.df[self.target_name].values

            # ---- 3. Split Data Training dan Testing (80:20) ----
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # ---- 4. Melatih Linear Regression ----
            X_train_lr = self.lr_scaler.fit_transform(self.X_train)
            X_test_lr = self.lr_scaler.transform(self.X_test)

            self.lr_model = LinearRegression()
            self.lr_model.fit(X_train_lr, self.y_train)
            self.lr_pred_test = self.lr_model.predict(X_test_lr)
            self.lr_metrics = self._calc_metrics(self.y_test, self.lr_pred_test)
            print("[INFO] Linear Regression berhasil dilatih!")

            # ---- 5. Melatih Random Forest ----
            X_train_rf = self.rf_scaler.fit_transform(self.X_train)
            X_test_rf = self.rf_scaler.transform(self.X_test)

            self.rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            self.rf_model.fit(X_train_rf, self.y_train)
            self.rf_pred_test = self.rf_model.predict(X_test_rf)
            self.rf_metrics = self._calc_metrics(self.y_test, self.rf_pred_test)
            print("[INFO] Random Forest berhasil dilatih!")

            self.is_trained = True

            return {
                'linear_regression': self.lr_metrics,
                'random_forest': self.rf_metrics
            }

        except Exception as e:
            print(f"[ERROR] Gagal melatih model: {str(e)}")
            raise e

    def _calc_metrics(self, y_true, y_pred):
        """
        Menghitung metrik evaluasi model:
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Squared Error)
        - R² Score (Koefisien Determinasi)
        """
        mse = mean_squared_error(y_true, y_pred)
        return {
            'mae': round(mean_absolute_error(y_true, y_pred), 4),
            'mse': round(mse, 4),
            'rmse': round(math.sqrt(mse), 4),
            'r2_score': round(r2_score(y_true, y_pred), 4),
            'train_size': len(self.y_train),
            'test_size': len(self.y_test),
            'total_data': len(self.y_train) + len(self.y_test)
        }

    def predict(self, input_data, model_type='linear_regression'):
        """
        Melakukan prediksi GDP Growth menggunakan model yang dipilih.
        
        Args:
            input_data (dict): Dictionary berisi nilai fitur
            model_type (str): 'linear_regression' atau 'random_forest'
                
        Returns:
            dict: Hasil prediksi beserta detail
        """
        if not self.is_trained:
            raise Exception("Model belum dilatih. Silakan muat dataset terlebih dahulu.")

        try:
            input_values = []
            for feature in self.feature_names:
                if feature in input_data:
                    input_values.append(float(input_data[feature]))
                else:
                    raise ValueError(f"Fitur '{feature}' tidak ditemukan dalam input")

            input_array = np.array(input_values).reshape(1, -1)

            if model_type == 'random_forest':
                input_scaled = self.rf_scaler.transform(input_array)
                prediction = self.rf_model.predict(input_scaled)[0]
                model_name = 'Random Forest'
            else:
                input_scaled = self.lr_scaler.transform(input_array)
                prediction = self.lr_model.predict(input_scaled)[0]
                model_name = 'Linear Regression'

            result = {
                'input': input_data,
                'prediction': round(float(prediction), 4),
                'unit': '% (Pertumbuhan GDP)',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model': model_name,
                'model_type': model_type
            }
            self.prediction_history.append(result)
            return result

        except Exception as e:
            raise Exception(f"Gagal melakukan prediksi: {str(e)}")

    def predict_comparison(self, input_data):
        """
        Melakukan prediksi GDP Growth menggunakan KEDUA model.
        Mengembalikan perbandingan hasil dan insight otomatis.
        
        Args:
            input_data (dict): Dictionary berisi nilai fitur
                
        Returns:
            dict: Hasil prediksi kedua model beserta insight
        """
        if not self.is_trained:
            raise Exception("Model belum dilatih.")

        try:
            input_values = []
            for feature in self.feature_names:
                if feature in input_data:
                    input_values.append(float(input_data[feature]))
                else:
                    raise ValueError(f"Fitur '{feature}' tidak ditemukan dalam input")

            input_array = np.array(input_values).reshape(1, -1)

            # Linear Regression prediction
            lr_scaled = self.lr_scaler.transform(input_array)
            lr_pred = float(self.lr_model.predict(lr_scaled)[0])

            # Random Forest prediction
            rf_scaled = self.rf_scaler.transform(input_array)
            rf_pred = float(self.rf_model.predict(rf_scaled)[0])

            # Determine best model based on R2 score
            lr_r2 = self.lr_metrics.get('r2_score', 0)
            rf_r2 = self.rf_metrics.get('r2_score', 0)
            lr_mae = self.lr_metrics.get('mae', 999)
            rf_mae = self.rf_metrics.get('mae', 999)

            if rf_r2 > lr_r2:
                best_model = 'Random Forest'
                best_pred = rf_pred
            else:
                best_model = 'Linear Regression'
                best_pred = lr_pred

            # Generate insight
            diff = abs(lr_pred - rf_pred)
            insight = self._generate_insight(lr_pred, rf_pred, best_model, diff, lr_r2, rf_r2, lr_mae, rf_mae)

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Store in history (best model prediction)
            result_record = {
                'input': input_data,
                'prediction': round(best_pred, 4),
                'unit': '% (Pertumbuhan GDP)',
                'timestamp': timestamp,
                'model': f'{best_model} (Best)'
            }
            self.prediction_history.append(result_record)

            return {
                'linear_regression': {
                    'prediction': round(lr_pred, 4),
                    'metrics': self.lr_metrics
                },
                'random_forest': {
                    'prediction': round(rf_pred, 4),
                    'metrics': self.rf_metrics
                },
                'best_model': best_model,
                'best_prediction': round(best_pred, 4),
                'difference': round(diff, 4),
                'insight': insight,
                'timestamp': timestamp,
                'input': input_data
            }

        except Exception as e:
            raise Exception(f"Gagal melakukan prediksi perbandingan: {str(e)}")

    def _generate_insight(self, lr_pred, rf_pred, best_model, diff, lr_r2, rf_r2, lr_mae, rf_mae):
        """Generate insight otomatis tentang perbandingan model."""
        insights = []

        # Insight tentang best model
        if best_model == 'Random Forest':
            insights.append(f"Random Forest menunjukkan performa lebih baik dengan R2 Score {rf_r2} vs {lr_r2} (Linear Regression).")
        else:
            insights.append(f"Linear Regression menunjukkan performa lebih baik dengan R2 Score {lr_r2} vs {rf_r2} (Random Forest).")

        # Insight tentang perbedaan prediksi
        if diff < 0.5:
            insights.append(f"Kedua model menghasilkan prediksi yang sangat mirip (selisih {round(diff, 2)}%), menunjukkan konsistensi yang baik.")
        elif diff < 1.5:
            insights.append(f"Terdapat perbedaan moderat ({round(diff, 2)}%) antara kedua model. Pertimbangkan menggunakan prediksi dari model terbaik.")
        else:
            insights.append(f"Perbedaan prediksi cukup signifikan ({round(diff, 2)}%). Disarankan menggunakan model dengan R2 Score tertinggi.")

        # Insight tentang MAE
        if rf_mae < lr_mae:
            insights.append(f"Random Forest memiliki error lebih rendah (MAE: {rf_mae}) dibanding Linear Regression (MAE: {lr_mae}).")
        else:
            insights.append(f"Linear Regression memiliki error lebih rendah (MAE: {lr_mae}) dibanding Random Forest (MAE: {rf_mae}).")

        # Insight tentang prediksi GDP
        avg_pred = (lr_pred + rf_pred) / 2
        if avg_pred > 5:
            insights.append("Prediksi menunjukkan pertumbuhan ekonomi yang kuat (>5%).")
        elif avg_pred > 3:
            insights.append("Prediksi menunjukkan pertumbuhan ekonomi moderat (3-5%).")
        elif avg_pred > 0:
            insights.append("Prediksi menunjukkan pertumbuhan ekonomi lambat (<3%).")
        else:
            insights.append("Prediksi menunjukkan potensi kontraksi ekonomi (pertumbuhan negatif).")

        return insights

    def get_metrics(self):
        """Mengembalikan metrik evaluasi dari model terbaik."""
        if not self.is_trained:
            return {}
            
        lr_r2 = self.lr_metrics.get('r2_score', 0)
        rf_r2 = self.rf_metrics.get('r2_score', 0)
        
        if rf_r2 > lr_r2:
            best_metrics = self.rf_metrics.copy()
            best_metrics['model_name'] = 'Random Forest'
        else:
            best_metrics = self.lr_metrics.copy()
            best_metrics['model_name'] = 'Linear Regression'
            
        return best_metrics

    def get_comparison_metrics(self):
        """Mengembalikan metrik perbandingan kedua model."""
        if not self.is_trained:
            return None

        lr_r2 = self.lr_metrics.get('r2_score', 0)
        rf_r2 = self.rf_metrics.get('r2_score', 0)
        lr_mae = self.lr_metrics.get('mae', 999)
        rf_mae = self.rf_metrics.get('mae', 999)

        best_model = 'Random Forest' if rf_r2 > lr_r2 else 'Linear Regression'

        return {
            'linear_regression': {
                'name': 'Linear Regression',
                'type': 'linear',
                'description': 'Model regresi linear yang mencari hubungan linear antara fitur dan target.',
                'metrics': self.lr_metrics,
                'actual_vs_predicted': {
                    'actual': [round(float(v), 4) for v in self.y_test],
                    'predicted': [round(float(v), 4) for v in self.lr_pred_test]
                }
            },
            'random_forest': {
                'name': 'Random Forest',
                'type': 'ensemble',
                'description': 'Model ensemble yang menggunakan banyak decision tree untuk prediksi yang lebih robust.',
                'metrics': self.rf_metrics,
                'actual_vs_predicted': {
                    'actual': [round(float(v), 4) for v in self.y_test],
                    'predicted': [round(float(v), 4) for v in self.rf_pred_test]
                },
                'feature_importance': {
                    name: round(float(imp), 6)
                    for name, imp in zip(self.feature_names, self.rf_model.feature_importances_)
                }
            },
            'best_model': best_model,
            'summary': self._generate_comparison_summary(best_model)
        }

    def _generate_comparison_summary(self, best_model):
        """Generate ringkasan perbandingan model."""
        lr = self.lr_metrics
        rf = self.rf_metrics
        return {
            'best_model': best_model,
            'r2_winner': 'Random Forest' if rf['r2_score'] > lr['r2_score'] else 'Linear Regression',
            'mae_winner': 'Random Forest' if rf['mae'] < lr['mae'] else 'Linear Regression',
            'rmse_winner': 'Random Forest' if rf['rmse'] < lr['rmse'] else 'Linear Regression',
            'r2_diff': round(abs(rf['r2_score'] - lr['r2_score']), 4),
            'mae_diff': round(abs(rf['mae'] - lr['mae']), 4),
            'rmse_diff': round(abs(rf['rmse'] - lr['rmse']), 4),
        }

    def get_history(self):
        """Mengembalikan riwayat prediksi."""
        return self.prediction_history

    def get_dataset_info(self):
        """
        Mengembalikan informasi dataset untuk ditampilkan di frontend.
        """
        if self.df is None:
            return None

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        return {
            'columns': self.df.columns.tolist(),
            'numeric_columns': numeric_cols,
            'shape': {'rows': self.df.shape[0], 'cols': self.df.shape[1]},
            'data': self.df.to_dict(orient='records'),
            'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            'statistics': json.loads(self.df.describe().to_json()),
            'feature_names': self.feature_names,
            'target': self.target_name,
            'time_cols': self.time_cols
        }

    def get_model_info(self, model_type='linear_regression'):
        """
        Mengembalikan informasi detail tentang model yang dipilih.
        """
        if not self.is_trained:
            return None

        if model_type == 'random_forest':
            feature_imp = {
                name: round(float(imp), 6)
                for name, imp in zip(self.feature_names, self.rf_model.feature_importances_)
            }
            return {
                'model_type': 'Random Forest',
                'library': 'scikit-learn',
                'features': self.feature_names,
                'target': self.target_name,
                'n_estimators': 100,
                'max_depth': 8,
                'feature_importance': feature_imp,
                'metrics': self.rf_metrics,
                'scaler': 'StandardScaler',
                'test_split': '20%',
                'train_split': '80%',
                'description': 'Model Random Forest menggunakan ensemble dari 100 decision tree untuk prediksi yang lebih robust terhadap non-linearitas data.',
                'actual_vs_predicted': {
                    'actual': [round(float(v), 4) for v in self.y_test],
                    'predicted': [round(float(v), 4) for v in self.rf_pred_test]
                }
            }
        else:
            coefficients = {}
            for name, coef in zip(self.feature_names, self.lr_model.coef_):
                coefficients[name] = round(float(coef), 6)

            return {
                'model_type': 'Linear Regression',
                'library': 'scikit-learn',
                'features': self.feature_names,
                'target': self.target_name,
                'coefficients': coefficients,
                'intercept': round(float(self.lr_model.intercept_), 6),
                'metrics': self.lr_metrics,
                'scaler': 'StandardScaler',
                'test_split': '20%',
                'train_split': '80%',
                'description': 'Model Linear Regression untuk memprediksi pertumbuhan GDP berdasarkan indikator ekonomi makro Indonesia.',
                'actual_vs_predicted': {
                    'actual': [round(float(v), 4) for v in self.y_test],
                    'predicted': [round(float(v), 4) for v in self.lr_pred_test]
                }
            }

    def get_trend_data(self):
        """
        Mengembalikan data tren secara dinamis berdasarkan dataset yang ada,
        menggunakan konfigurasi kolom waktu jika tersedia.
        """
        if self.df is None:
            return None

        labels = []
        has_time_config = self.time_cols and any(self.time_cols.values())
        
        for i in range(len(self.df)):
            if has_time_config:
                label_parts = []
                if self.time_cols.get('year') and self.time_cols['year'] in self.df.columns:
                    # Coba format sebagai int untuk menghilangkan .0 jika tahun berbentuk float
                    try:
                        yr = int(float(self.df.iloc[i][self.time_cols['year']]))
                        label_parts.append(str(yr))
                    except:
                        label_parts.append(str(self.df.iloc[i][self.time_cols['year']]))
                
                if self.time_cols.get('month') and self.time_cols['month'] in self.df.columns:
                    label_parts.append(str(self.df.iloc[i][self.time_cols['month']]))
                
                if self.time_cols.get('quarter') and self.time_cols['quarter'] in self.df.columns:
                    try:
                        q = int(float(self.df.iloc[i][self.time_cols['quarter']]))
                        label_parts.append(f"Q{q}")
                    except:
                        label_parts.append(f"Q{self.df.iloc[i][self.time_cols['quarter']]}")
                
                if label_parts:
                    labels.append(" ".join(label_parts))
                else:
                    labels.append(str(self.df.iloc[i][self.df.columns[0]]))
            else:
                # Gunakan kolom pertama sebagai default jika tidak ada konfigurasi
                first_col = self.df.columns[0]
                labels.append(str(self.df.iloc[i][first_col]))

        # Gunakan fitur-fitur untuk grafik indikator
        indicators = {}
        for col in self.feature_names:
            indicators[col] = self.df[col].tolist()

        return {
            'labels': labels,
            'target_name': self.target_name,
            'target_data': self.df[self.target_name].tolist(),
            'indicators': indicators
        }

    def clear_history(self):
        """Menghapus riwayat prediksi."""
        self.prediction_history = []

    def get_eda(self):
        """
        Melakukan Exploratory Data Analysis dan mengembalikan data 
        untuk divisualisasikan di frontend.
        """
        if self.df is None:
            return None
        
        # Buat dataframe khusus untuk EDA yang hanya berisi fitur terpilih dan target
        selected_cols = list(dict.fromkeys(self.feature_names + [self.target_name]))
        df_eda = self.df[selected_cols]

        # 1. Descriptive stats (hanya numerik)
        desc = df_eda.describe(include=[np.number]).to_dict()
        
        # 2. Correlation Matrix (hanya numerik)
        corr_matrix = df_eda.corr(numeric_only=True)
        corr_dict = corr_matrix.round(3).to_dict()

        # 3. Missing values
        missing_values = df_eda.isnull().sum().to_dict()

        # 4. Outlier detection (IQR method) & Skewness
        outliers_info = {}
        skewness_info = {}
        for col in df_eda.select_dtypes(include=[np.number]).columns:
            Q1 = df_eda[col].quantile(0.25)
            Q3 = df_eda[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df_eda[(df_eda[col] < lower_bound) | (df_eda[col] > upper_bound)]
            outliers_info[col] = len(outliers)
            skewness_info[col] = df_eda[col].skew()

        # 5. Generate Insights
        insights = []
        
        # Insight: Highest correlations (excluding self and target)
        highest_corr = 0
        corr_pair = ("", "")
        features_only = [c for c in corr_matrix.columns if c != self.target_name]
        for i in range(len(features_only)):
            for j in range(i+1, len(features_only)):
                f1, f2 = features_only[i], features_only[j]
                val = abs(corr_matrix.loc[f1, f2])
                if val > highest_corr:
                    highest_corr = val
                    corr_pair = (f1, f2)
        
        if highest_corr > 0.8:
            insights.append(f"Potensi Multikolinearitas: {corr_pair[0]} dan {corr_pair[1]} memiliki korelasi sangat kuat ({highest_corr:.2f}).")
        elif highest_corr > 0.6:
            insights.append(f"Korelasi tertinggi antar fitur: {corr_pair[0]} dan {corr_pair[1]} ({highest_corr:.2f}).")
            
        # Insight: Correlation with target
        if self.target_name in corr_matrix.columns:
            target_corr = corr_matrix[self.target_name].drop(self.target_name).abs().sort_values(ascending=False)
            top_feature = target_corr.index[0]
            insights.append(f"Fitur '{top_feature}' memiliki korelasi terkuat dengan target '{self.target_name}' ({corr_matrix.loc[top_feature, self.target_name]:.2f}).")

        # Insight: Outliers
        total_outliers = sum(outliers_info.values())
        if total_outliers > 0:
            cols_with_outliers = [k for k, v in outliers_info.items() if v > 0]
            insights.append(f"Ditemukan {total_outliers} potensi outlier pada variabel: {', '.join(cols_with_outliers)}.")
        else:
            insights.append("Tidak ditemukan outlier signifikan menggunakan metode IQR.")

        # Insight: Skewness
        skewed_cols = [k for k, v in skewness_info.items() if abs(v) > 1]
        if skewed_cols:
            insights.append(f"Distribusi tidak normal (skewed) terdeteksi pada: {', '.join(skewed_cols)}.")

        # 6. Scatter data preparation (sampling to avoid large payload if dataset is huge, but here it's 40 rows)
        # We can just pass the dataframe records
        scatter_data = df_eda.to_dict(orient='records')

        return {
            'descriptive': desc,
            'correlation': corr_dict,
            'missing_values': missing_values,
            'outliers': outliers_info,
            'skewness': skewness_info,
            'insights': insights,
            'scatter_data': scatter_data,
            'columns': list(df_eda.columns),
            'numeric_columns': df_eda.select_dtypes(include=[np.number]).columns.tolist()
        }

