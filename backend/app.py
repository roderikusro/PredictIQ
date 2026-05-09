"""
=============================================================
  Backend Flask - Sistem Statistik Prediktif
  API Server untuk Machine Learning & Data Management
=============================================================
  Endpoint API:
  - GET  /                    → Halaman utama (SPA)
  - GET  /api/dashboard       → Data dashboard
  - GET  /api/dataset         → Data dataset
  - GET  /api/model-info      → Informasi model ML
  - GET  /api/trend           → Data tren untuk grafik
  - GET  /api/history         → Riwayat prediksi
  - POST /api/predict         → Melakukan prediksi
  - POST /api/upload-csv      → Upload dataset CSV baru
  - GET  /api/export          → Export riwayat prediksi
  - POST /api/clear-history   → Hapus riwayat prediksi
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
    metrics = model.get_metrics()
    print(f"  MAE: {metrics['mae']}")
    print(f"  MSE: {metrics['mse']}")
    print(f"  R² Score: {metrics['r2_score']}")
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
    Termasuk koefisien, metrik, dan konfigurasi.
    """
    try:
        model_info = model.get_model_info()
        if model_info:
            return jsonify({'status': 'success', 'data': model_info})
        else:
            return jsonify({'status': 'error', 'message': 'Model belum dilatih'}), 404

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
    
    Request Body (JSON):
    {
        "Tahun": 2025,
        "Kuartal": 1,
        "Populasi_Juta": 280,
        "Inflasi_Persen": 3.0,
        "Suku_Bunga_Persen": 5.5,
        "Pengangguran_Persen": 5.0,
        "Investasi_Triliun": 275,
        "Ekspor_Miliar_USD": 58,
        "Konsumsi_RT_Triliun": 1950
    }
    """
    try:
        input_data = request.get_json()

        if not input_data:
            return jsonify({'status': 'error', 'message': 'Data input kosong'}), 400

        # Validasi semua fitur ada
        missing = [f for f in model.feature_names if f not in input_data]
        if missing:
            return jsonify({
                'status': 'error',
                'message': f'Fitur yang kurang: {", ".join(missing)}',
                'required_features': model.feature_names
            }), 400

        # Lakukan prediksi
        result = model.predict(input_data)
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
    Upload file CSV baru untuk melatih ulang model.
    File harus memiliki kolom 'GDP_Growth_Persen' sebagai target.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'Tidak ada file yang diunggah'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'Nama file kosong'}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({'status': 'error', 'message': 'Hanya file CSV yang diizinkan'}), 400

        # Simpan file yang diunggah
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"uploaded_{timestamp}.csv"
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)

        # Latih ulang model dengan dataset baru
        global model
        model = PredictiveModel(dataset_path=filepath)

        return jsonify({
            'status': 'success',
            'message': 'Dataset berhasil diunggah dan model dilatih ulang',
            'filename': filename,
            'metrics': model.get_metrics()
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/export', methods=['GET'])
def api_export():
    """
    Export riwayat prediksi dalam format CSV.
    """
    try:
        history = model.get_history()

        if not history:
            return jsonify({'status': 'error', 'message': 'Belum ada riwayat prediksi'}), 404

        # Buat CSV dari riwayat
        output = StringIO()
        writer = csv.writer(output)

        # Header
        header = ['No', 'Timestamp', 'Prediksi_GDP_Growth', 'Model']
        input_keys = list(history[0]['input'].keys()) if history else []
        header.extend(input_keys)
        writer.writerow(header)

        # Data
        for i, record in enumerate(history, 1):
            row = [i, record['timestamp'], record['prediction'], record['model']]
            for key in input_keys:
                row.append(record['input'].get(key, ''))
            writer.writerow(row)

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
