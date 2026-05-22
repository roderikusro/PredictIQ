"""
=============================================================
  Backend Flask - Sistem Statistik Prediktif
  API Server untuk Machine Learning & Data Management
=============================================================
  Endpoint API:
  - GET  /                        -> Halaman utama (SPA)
  - GET  /api/dashboard           -> Data dashboard
  - GET  /api/dataset             -> Data dataset
  - GET  /api/model-info          -> Informasi model ML
  - GET  /api/trend               -> Data tren untuk grafik
  - GET  /api/history             -> Riwayat prediksi
  - POST /api/predict             -> Melakukan prediksi (LR)
  - POST /api/predict-compare     -> Prediksi perbandingan
  - GET  /api/model-comparison    -> Data perbandingan model
  - POST /api/upload-csv          -> Upload dataset CSV baru
  - GET  /api/export              -> Export riwayat prediksi
  - POST /api/clear-history       -> Hapus riwayat prediksi
  - GET  /api/infographic-data    -> Data lengkap untuk infografis
=============================================================
"""

import os
import sys
import json
import csv
from io import StringIO
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_cors import CORS

# Tambahkan path backend ke sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_model import PredictiveModel

# ---- Konfigurasi Flask ----
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'),
    static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
)
CORS(app)  # Aktifkan CORS untuk development

# ---- Konfigurasi Path ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
DEFAULT_DATASET = os.path.join(DATASET_DIR, 'ekonomi_data.csv')

# Buat folder uploads jika belum ada
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---- Inisialisasi Model ML ----
print("=" * 60)
print("  Sistem Statistik Prediktif - Inisialisasi")
print("=" * 60)
model = PredictiveModel(dataset_path=DEFAULT_DATASET)
print(f"  Model Status: {'Aktif [OK]' if model.is_trained else 'Belum Dilatih [X]'}")
if model.is_trained:
    lr_m = model.lr_metrics
    rf_m = model.rf_metrics
    print(f"  --- Linear Regression ---")
    print(f"  MAE: {lr_m['mae']}  |  RMSE: {lr_m['rmse']}  |  R2: {lr_m['r2_score']}")
    print(f"  --- Random Forest ---")
    print(f"  MAE: {rf_m['mae']}  |  RMSE: {rf_m['rmse']}  |  R2: {rf_m['r2_score']}")
print("=" * 60)


# ===========================================================
#                    ROUTE HALAMAN
# ===========================================================

@app.route('/')
def index():
    """Render halaman utama (Single Page Application)."""
    return render_template('index.html')


# ===========================================================
#                    API ENDPOINTS
# ===========================================================

@app.route('/api/dashboard', methods=['GET'])
def api_dashboard():
    """
    Mengembalikan data ringkasan untuk dashboard.
    Termasuk statistik model, jumlah data, dan prediksi terakhir.
    """
    try:
        metrics = model.get_metrics()
        dataset_info = model.get_dataset_info()
        history = model.get_history()

        dashboard_data = {
            'status': 'success',
            'model_trained': model.is_trained,
            'metrics': metrics,
            'dataset_rows': dataset_info['shape']['rows'] if dataset_info else 0,
            'dataset_cols': dataset_info['shape']['cols'] if dataset_info else 0,
            'total_predictions': len(history),
            'last_prediction': history[-1] if history else None,
            'features': model.feature_names,
            'target_name': model.target_name,
            'time_cols': getattr(model, 'time_cols', {}),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return jsonify(dashboard_data)

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/dataset', methods=['GET'])
def api_dataset():
    """
    Mengembalikan data dataset lengkap untuk ditampilkan di tabel.
    """
    try:
        dataset_info = model.get_dataset_info()
        if dataset_info:
            return jsonify({'status': 'success', 'data': dataset_info})
        else:
            return jsonify({'status': 'error', 'message': 'Dataset belum dimuat'}), 404

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/model-info', methods=['GET'])
def api_model_info():
    """
    Mengembalikan informasi detail tentang model ML.
    Query param: ?type=linear_regression atau ?type=random_forest
    """
    try:
        model_type = request.args.get('type', 'linear_regression')
        model_info = model.get_model_info(model_type=model_type)
        if model_info:
            return jsonify({'status': 'success', 'data': model_info})
        else:
            return jsonify({'status': 'error', 'message': 'Model belum dilatih'}), 404

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/model-comparison', methods=['GET'])
def api_model_comparison():
    """
    Mengembalikan data perbandingan performa kedua model.
    Termasuk metrik, actual vs predicted, feature importance, dan insight.
    """
    try:
        comparison = model.get_comparison_metrics()
        if comparison:
            return jsonify({'status': 'success', 'data': comparison})
        else:
            return jsonify({'status': 'error', 'message': 'Model belum dilatih'}), 404

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/eda', methods=['GET'])
def api_eda():
    """
    Mengembalikan data Exploratory Data Analysis (EDA).
    Termasuk korelasi, outlier, skewness, descriptive stats, dan insight.
    """
    try:
        remove_outliers = request.args.get('remove_outliers', 'false').lower() == 'true'
        eda_data = model.get_eda(remove_outliers=remove_outliers)
        if eda_data:
            return jsonify({'status': 'success', 'data': eda_data})
        else:
            return jsonify({'status': 'error', 'message': 'Dataset belum dimuat'}), 404

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/export-eda', methods=['GET'])
def api_export_eda():
    """
    Export tabel statistik deskriptif EDA dalam format CSV atau Excel.
    """
    try:
        format_type = request.args.get('format', 'csv')
        remove_outliers = request.args.get('remove_outliers', 'false').lower() == 'true'
        eda_data = model.get_eda(remove_outliers=remove_outliers)
        if not eda_data:
            return jsonify({'status': 'error', 'message': 'Dataset belum dimuat'}), 404
            
        header = ['Fitur', 'Obs', 'Mean', 'Median (50%)', 'Std Dev', 'Min', 'Max', 'Outliers (IQR)', 'Missing']
        data = []
        num_cols = eda_data.get('numeric_columns', [])
        
        import pandas as pd
        for col in num_cols:
            desc = eda_data['descriptive'].get(col)
            out = eda_data['outliers'].get(col, 0)
            mis = eda_data['missing_values'].get(col, 0)
            if not desc: continue
            
            row = [
                col,
                int(desc.get('count', 0)) if pd.notnull(desc.get('count')) else '-',
                round(desc.get('mean', 0), 2) if pd.notnull(desc.get('mean')) else '-',
                round(desc.get('50%', 0), 2) if pd.notnull(desc.get('50%')) else '-',
                round(desc.get('std', 0), 2) if pd.notnull(desc.get('std')) else '-',
                round(desc.get('min', 0), 2) if pd.notnull(desc.get('min')) else '-',
                round(desc.get('max', 0), 2) if pd.notnull(desc.get('max')) else '-',
                out,
                mis
            ]
            data.append(row)
            
        if format_type == 'excel':
            import io
            df = pd.DataFrame(data, columns=header)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Statistik Deskriptif')
            output.seek(0)
            return Response(
                output.getvalue(),
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                headers={'Content-Disposition': f'attachment; filename=eda_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'}
            )
        else:
            output = StringIO()
            writer = csv.writer(output)
            writer.writerow(header)
            writer.writerows(data)
            output.seek(0)
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename=eda_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'}
            )
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/trend', methods=['GET'])
def api_trend():
    """
    Mengembalikan data tren untuk visualisasi grafik.
    """
    try:
        trend_data = model.get_trend_data()
        if trend_data:
            return jsonify({'status': 'success', 'data': trend_data})
        else:
            return jsonify({'status': 'error', 'message': 'Data tren tidak tersedia'}), 404

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Melakukan prediksi berdasarkan input pengguna.
    Mendukung pemilihan model via field 'model_type'.
    """
    try:
        input_data = request.get_json()

        if not input_data:
            return jsonify({'status': 'error', 'message': 'Data input kosong'}), 400

        # Ambil model_type lalu hapus dari input agar tidak masuk ke fitur
        model_type = input_data.pop('model_type', 'linear_regression')

        # Validasi semua fitur ada
        missing = [f for f in model.feature_names if f not in input_data]
        if missing:
            return jsonify({
                'status': 'error',
                'message': f'Fitur yang kurang: {", ".join(missing)}',
                'required_features': model.feature_names
            }), 400

        # Lakukan prediksi dengan model yang dipilih
        result = model.predict(input_data, model_type=model_type)
        return jsonify({'status': 'success', 'data': result})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/predict-compare', methods=['POST'])
def api_predict_compare():
    """
    Melakukan prediksi menggunakan KEDUA model sekaligus.
    Mengembalikan hasil perbandingan dan insight otomatis.
    """
    try:
        input_data = request.get_json()

        if not input_data:
            return jsonify({'status': 'error', 'message': 'Data input kosong'}), 400

        missing = [f for f in model.feature_names if f not in input_data]
        if missing:
            return jsonify({
                'status': 'error',
                'message': f'Fitur yang kurang: {", ".join(missing)}',
                'required_features': model.feature_names
            }), 400

        result = model.predict_comparison(input_data)
        return jsonify({'status': 'success', 'data': result})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def api_history():
    """
    Mengembalikan riwayat semua prediksi yang telah dilakukan.
    """
    try:
        history = model.get_history()
        return jsonify({
            'status': 'success',
            'data': history,
            'total': len(history)
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/upload-csv', methods=['POST'])
def api_upload_csv():
    """
    Upload file CSV atau Excel baru untuk melatih ulang model.
    File harus memiliki kolom 'GDP_Growth_Persen' sebagai target.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'Tidak ada file yang diunggah'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'Nama file kosong'}), 400

        if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
            return jsonify({'status': 'error', 'message': 'Hanya file CSV atau Excel (.xlsx, .xls) yang diizinkan'}), 400

        # Simpan file yang diunggah
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"uploaded_{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        file.save(file_path)

        # Latih model dengan dataset baru
        metrics = model.load_and_train(file_path)
        
        return jsonify({
            'status': 'success',
            'message': 'Dataset berhasil diunggah dan model telah dilatih',
            'filename': filename,
            'metrics': metrics
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/configure-model', methods=['POST'])
def api_configure_model():
    """
    Mengonfigurasi ulang model dengan memilih fitur dan target baru,
    kemudian melatih ulang model.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'Data konfigurasi kosong'}), 400

        target_col = data.get('target')
        feature_cols = data.get('features')
        time_cols = data.get('time_cols')

        if not target_col or not feature_cols:
            return jsonify({'status': 'error', 'message': 'Target dan fitur harus dipilih'}), 400

        if len(feature_cols) < 1:
            return jsonify({'status': 'error', 'message': 'Minimal pilih 1 fitur'}), 400

        # Latih ulang model
        metrics = model.load_and_train(target_col=target_col, feature_cols=feature_cols, time_cols=time_cols)

        return jsonify({
            'status': 'success',
            'message': 'Model berhasil dikonfigurasi dan dilatih ulang',
            'metrics': metrics
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/export', methods=['GET'])
def api_export():
    """
    Export riwayat prediksi dalam format CSV atau Excel.
    """
    try:
        format_type = request.args.get('format', 'csv')
        history = model.get_history()

        if not history:
            return jsonify({'status': 'error', 'message': 'Belum ada riwayat prediksi'}), 404

        # Menyiapkan data list
        header = ['No', 'Timestamp', 'Prediksi_GDP_Growth', 'Model']
        input_keys = list(history[0]['input'].keys()) if history else []
        header.extend(input_keys)
        
        data = []
        for i, record in enumerate(history, 1):
            row = [i, record['timestamp'], record['prediction'], record['model']]
            for key in input_keys:
                row.append(record['input'].get(key, ''))
            data.append(row)

        if format_type == 'excel':
            import io
            import pandas as pd
            
            df = pd.DataFrame(data, columns=header)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Riwayat Prediksi')
            
            output.seek(0)
            return Response(
                output.getvalue(),
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                headers={
                    'Content-Disposition': f'attachment; filename=prediksi_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
                }
            )
        else:
            # Default: Buat CSV
            output = StringIO()
            writer = csv.writer(output)
            writer.writerow(header)
            writer.writerows(data)

            # Kirim sebagai file download
            output.seek(0)
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={
                    'Content-Disposition': f'attachment; filename=prediksi_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                }
            )

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/infographic-data', methods=['GET'])
def api_infographic_data():
    """
    Mengembalikan data lengkap untuk halaman Infografis Otomatis.
    Menggabungkan dashboard, EDA, model comparison, dan trend menjadi satu payload.
    """
    try:
        if not model.is_trained:
            return jsonify({'status': 'error', 'message': 'Model belum dilatih'}), 404

        # --- 1. Dataset Overview ---
        dataset_info = model.get_dataset_info()
        overview = {
            'rows': dataset_info['shape']['rows'] if dataset_info else 0,
            'cols': dataset_info['shape']['cols'] if dataset_info else 0,
            'features': model.feature_names,
            'target': model.target_name,
        }

        # --- 2. Model Metrics (Best Model) ---
        metrics = model.get_metrics()

        # --- 3. EDA Highlights ---
        eda_data = model.get_eda()
        eda_highlights = {}
        if eda_data:
            eda_highlights['insights'] = eda_data.get('insights', [])
            eda_highlights['outliers'] = eda_data.get('outliers', {})
            eda_highlights['skewness'] = eda_data.get('skewness', {})
            eda_highlights['forecast'] = eda_data.get('forecast')
            # Top 3 correlations with target
            if model.target_name in eda_data.get('correlation', {}):
                target_corr = eda_data['correlation'][model.target_name]
                sorted_corr = sorted(
                    [(k, v) for k, v in target_corr.items() if k != model.target_name and v is not None],
                    key=lambda x: abs(x[1]), reverse=True
                )
                eda_highlights['top_correlations'] = [
                    {'feature': k, 'correlation': v} for k, v in sorted_corr[:5]
                ]
            # Descriptive stats summary (for each feature: mean, min, max)
            desc_summary = []
            for col in model.feature_names:
                desc = eda_data.get('descriptive', {}).get(col, {})
                if desc:
                    desc_summary.append({
                        'feature': col,
                        'mean': round(desc.get('mean', 0), 2) if desc.get('mean') is not None else None,
                        'min': round(desc.get('min', 0), 2) if desc.get('min') is not None else None,
                        'max': round(desc.get('max', 0), 2) if desc.get('max') is not None else None,
                        'std': round(desc.get('std', 0), 2) if desc.get('std') is not None else None
                    })
            eda_highlights['descriptive_summary'] = desc_summary
            eda_highlights['scatter_data'] = eda_data.get('scatter_data', [])

        # --- 4. Model Comparison ---
        comparison = model.get_comparison_metrics()
        comparison_summary = {}
        if comparison:
            comparison_summary = {
                'best_model': comparison['best_model'],
                'lr_metrics': comparison['linear_regression']['metrics'],
                'rf_metrics': comparison['random_forest']['metrics'],
                'summary': comparison['summary'],
            }
            # Include feature importance from RF
            if 'feature_importance' in comparison.get('random_forest', {}):
                comparison_summary['feature_importance'] = comparison['random_forest']['feature_importance']

        # --- 5. Trend Data (for sparkline/mini chart) ---
        trend = model.get_trend_data()
        trend_summary = {}
        if trend:
            target_data = trend.get('target_data', [])
            forecast = trend.get('forecast')
            trend_summary = {
                'labels': trend.get('labels', []),
                'target_data': target_data,
                'target_name': trend.get('target_name', ''),
                'latest_value': target_data[-1] if target_data else None,
                'avg_value': round(sum(target_data) / len(target_data), 2) if target_data else None,
                'min_value': round(min(target_data), 2) if target_data else None,
                'max_value': round(max(target_data), 2) if target_data else None,
                'forecast': forecast,
                'forecast_labels': forecast.get('labels', []) if forecast else [],
                'forecast_data': forecast.get('target_data', []) if forecast else [],
                'forecast_latest': forecast.get('target_data', [None])[-1] if forecast and forecast.get('target_data') else None,
                'forecast_insight': forecast.get('insight') if forecast else None,
            }

        # --- 6. Auto-generated narrative paragraphs ---
        narratives = []
        # Narrative 1: Dataset overview
        narratives.append(f"Dataset berisi {overview['rows']} baris data dan {overview['cols']} kolom. Target prediksi adalah '{model.target_name}' dengan {len(model.feature_names)} fitur pendukung.")
        # Narrative 2: Model performance
        if metrics:
            r2 = metrics.get('r2_score', 0)
            model_name = metrics.get('model_name', 'Model')
            if r2 > 0.9:
                perf = 'sangat baik'
            elif r2 > 0.7:
                perf = 'baik'
            elif r2 > 0.5:
                perf = 'cukup'
            else:
                perf = 'perlu ditingkatkan'
            narratives.append(f"Model terbaik ({model_name}) menunjukkan performa {perf} dengan R² Score {r2}, MAE {metrics.get('mae', '-')}, dan RMSE {metrics.get('rmse', '-')}.")
        # Narrative 3: Trend insight
        if trend_summary.get('target_data') and len(trend_summary['target_data']) >= 2:
            last = trend_summary['target_data'][-1]
            prev = trend_summary['target_data'][-2]
            if last > prev:
                narratives.append(f"Tren terbaru menunjukkan kenaikan {model.target_name} dari {round(prev,2)} ke {round(last,2)}.")
            elif last < prev:
                narratives.append(f"Tren terbaru menunjukkan penurunan {model.target_name} dari {round(prev,2)} ke {round(last,2)}.")
            else:
                narratives.append(f"{model.target_name} stabil di angka {round(last,2)}.")
        if trend_summary.get('forecast_insight'):
            narratives.append(trend_summary['forecast_insight'])
        # Narrative 4: Key features
        if comparison_summary.get('feature_importance'):
            sorted_fi = sorted(comparison_summary['feature_importance'].items(), key=lambda x: x[1], reverse=True)
            top_features = [f[0] for f in sorted_fi[:3]]
            narratives.append(f"Fitur yang paling berpengaruh terhadap prediksi: {', '.join(top_features)}.")

        return jsonify({
            'status': 'success',
            'data': {
                'overview': overview,
                'metrics': metrics,
                'eda_highlights': eda_highlights,
                'comparison': comparison_summary,
                'trend': trend_summary,
                'narratives': narratives,
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/clear-history', methods=['POST'])
def api_clear_history():
    """Menghapus semua riwayat prediksi."""
    try:
        model.clear_history()
        return jsonify({'status': 'success', 'message': 'Riwayat prediksi berhasil dihapus'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ===========================================================
#                    MENJALANKAN SERVER
# ===========================================================

if __name__ == '__main__':
    print("\n>> Server berjalan di http://localhost:5000")
    print(">> Sistem Statistik Prediktif siap digunakan!\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
