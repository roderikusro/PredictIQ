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
import uuid
from io import StringIO
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, Response, session
from flask_cors import CORS
from functools import wraps

# Tambahkan path backend ke sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_model import PredictiveModel, XGBOOST_AVAILABLE
from database import (
    init_db, register_user, login_user, count_users,
    get_user_active_dataset, set_user_active_dataset,
    save_prediction, get_user_history, get_all_history_admin,
    soft_delete_prediction, hard_delete_prediction_admin, clear_all_history
)

# ---- Konfigurasi Flask ----
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'),
    static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
)
app.secret_key = 'predictiq_super_secret_session_key_2026'
CORS(app, supports_credentials=True)  # Aktifkan CORS untuk development dengan kredensial sesi

# Inisialisasi Database SQLite
init_db()

# ---- Decorator Keamanan ----
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return jsonify({'status': 'error', 'message': 'Autentikasi diperlukan. Silakan login.'}), 401
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return jsonify({'status': 'error', 'message': 'Akses Admin diperlukan.'}), 403
        return f(*args, **kwargs)
    return decorated_function

# ---- Registri Model Per Sesi ----
user_models = {}

def get_session_id():
    """Mengambil username aktif sebagai ID sesi, atau UUID default jika belum login."""
    if 'username' in session:
        return session['username']
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

def get_user_model():
    """Mengambil instance PredictiveModel terisolasi untuk sesi saat ini."""
    sid = get_session_id()
    if sid not in user_models:
        # Periksa apakah ada dataset aktif yang tersimpan untuk user ini di DB
        active_ds = None
        if 'username' in session:
            active_ds = get_user_active_dataset(session['username'])
        
        # Inisialisasi PredictiveModel
        if active_ds and os.path.exists(active_ds):
            user_models[sid] = PredictiveModel(dataset_path=active_ds)
            print(f"[INFO] Model untuk user '{sid}' dimuat otomatis menggunakan dataset aktif: {os.path.basename(active_ds)}")
        else:
            # Inisialisasi awal kosong (belum dilatih) sesuai request user
            user_models[sid] = PredictiveModel(dataset_path=None)
    return user_models[sid]


# ---- Konfigurasi Path ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
DEFAULT_DATASET = os.path.join(DATASET_DIR, 'ekonomi_data.csv')
VALID_MODEL_TYPES = ['linear_regression', 'random_forest', 'xgboost']

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
    if XGBOOST_AVAILABLE and model.xgb_metrics:
        xgb_m = model.xgb_metrics
        print(f"  --- XGBoost ---")
        print(f"  MAE: {xgb_m['mae']}  |  RMSE: {xgb_m['rmse']}  |  R2: {xgb_m['r2_score']}")
    else:
        print("  --- XGBoost ---")
        print("  Tidak tersedia atau belum dilatih")
print("=" * 60)


# ===========================================================
#                    ROUTE HALAMAN
# ===========================================================

@app.route('/')
def index():
    """Render halaman utama (Single Page Application)."""
    get_session_id()  # Inisialisasi sesi saat pertama kali mengakses
    return render_template('index.html')


# ===========================================================
#                    AUTHENTICATION ENDPOINTS
# ===========================================================

@app.route('/api/auth/register', methods=['POST'])
def api_auth_register():
    try:
        data = request.get_json()
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({'status': 'error', 'message': 'Parameter tidak lengkap'}), 400
        success, msg = register_user(data['username'], data['password'])
        if success:
            return jsonify({'status': 'success', 'message': msg})
        return jsonify({'status': 'error', 'message': msg}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def api_auth_login():
    try:
        data = request.get_json()
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({'status': 'error', 'message': 'Parameter tidak lengkap'}), 400
        success, res = login_user(data['username'], data['password'])
        if success:
            session['username'] = res
            return jsonify({'status': 'success', 'message': 'Login berhasil', 'username': res})
        return jsonify({'status': 'error', 'message': res}), 401
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/auth/logout', methods=['POST'])
def api_auth_logout():
    session.pop('username', None)
    session.pop('admin_logged_in', None)
    return jsonify({'status': 'success', 'message': 'Logout berhasil'})

@app.route('/api/auth/status', methods=['GET'])
def api_auth_status():
    logged_in = 'username' in session
    total_users = count_users()
    return jsonify({
        'status': 'success',
        'logged_in': logged_in,
        'username': session.get('username'),
        'total_users': total_users
    })

@app.route('/api/admin/login', methods=['POST'])
def api_admin_login():
    try:
        data = request.get_json()
        if not data or 'password' not in data:
            return jsonify({'status': 'error', 'message': 'Password admin diperlukan'}), 400
        
        if data['password'] == 'admin123':
            session['admin_logged_in'] = True
            return jsonify({'status': 'success', 'message': 'Login admin berhasil'})
        return jsonify({'status': 'error', 'message': 'Password admin salah'}), 401
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ===========================================================
#                    API ENDPOINTS
# ===========================================================

@app.route('/api/dashboard', methods=['GET'])
@login_required
def api_dashboard():
    """
    Mengembalikan data ringkasan untuk dashboard.
    Termasuk statistik model, jumlah data, dan prediksi terakhir.
    """
    try:
        model = get_user_model()
        metrics = model.get_metrics()
        dataset_info = model.get_dataset_info()
        history = get_user_history(session['username'])

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
@login_required
def api_dataset():
    """
    Mengembalikan data dataset lengkap untuk ditampilkan di tabel.
    """
    try:
        model = get_user_model()
        dataset_info = model.get_dataset_info()
        if dataset_info:
            return jsonify({'status': 'success', 'data': dataset_info})
        else:
            return jsonify({'status': 'success', 'data': None, 'message': 'Dataset belum dimuat'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/model-info', methods=['GET'])
@login_required
def api_model_info():
    """
    Mengembalikan informasi detail tentang model ML.
    Query param: ?type=linear_regression atau ?type=random_forest
    """
    try:
        model = get_user_model()
        model_type = request.args.get('type', 'linear_regression')
        if model_type not in VALID_MODEL_TYPES:
            return jsonify({
                'status': 'error',
                'message': f'model_type harus salah satu dari: {VALID_MODEL_TYPES}'
            }), 400

        model_info = model.get_model_info(model_type=model_type)
        if model_info:
            return jsonify({'status': 'success', 'data': model_info})
        else:
            return jsonify({'status': 'error', 'message': 'Model belum dilatih'}), 404

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/model-comparison', methods=['GET'])
@login_required
def api_model_comparison():
    """
    Mengembalikan data perbandingan performa kedua model.
    Termasuk metrik, actual vs predicted, feature importance, dan insight.
    """
    try:
        model = get_user_model()
        comparison = model.get_comparison_metrics()
        if comparison:
            return jsonify({'status': 'success', 'data': comparison})
        else:
            return jsonify({'status': 'error', 'message': 'Model belum dilatih'}), 404

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/eda', methods=['GET'])
@login_required
def api_eda():
    """
    Mengembalikan data Exploratory Data Analysis (EDA).
    Termasuk korelasi, outlier, skewness, descriptive stats, dan insight.
    """
    try:
        model = get_user_model()
        remove_outliers = request.args.get('remove_outliers', 'false').lower() == 'true'
        eda_data = model.get_eda(remove_outliers=remove_outliers)
        if eda_data:
            return jsonify({'status': 'success', 'data': eda_data})
        else:
            return jsonify({'status': 'error', 'message': 'Dataset belum dimuat'}), 404

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/export-eda', methods=['GET'])
@login_required
def api_export_eda():
    """
    Export tabel statistik deskriptif EDA dalam format CSV atau Excel.
    """
    try:
        model = get_user_model()
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
@login_required
def api_trend():
    """
    Mengembalikan data tren untuk visualisasi grafik.
    """
    try:
        model = get_user_model()
        trend_data = model.get_trend_data()
        if trend_data:
            return jsonify({'status': 'success', 'data': trend_data})
        else:
            return jsonify({'status': 'error', 'message': 'Data tren tidak tersedia'}), 404

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """
    Melakukan prediksi berdasarkan input pengguna.
    Mendukung pemilihan model via field 'model_type'.
    """
    try:
        model = get_user_model()
        input_data = request.get_json()

        if not input_data:
            return jsonify({'status': 'error', 'message': 'Data input kosong'}), 400

        # Ambil model_type lalu hapus dari input agar tidak masuk ke fitur
        model_type = input_data.pop('model_type', 'linear_regression')
        if model_type not in VALID_MODEL_TYPES:
            return jsonify({
                'status': 'error',
                'message': f'model_type harus salah satu dari: {VALID_MODEL_TYPES}'
            }), 400

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

        # Simpan ke database SQLite
        username = session['username']
        save_prediction(
            username=username,
            model_name=result['model'],
            model_type=result['model_type'],
            input_data=result['input'],
            prediction_value=result['prediction'],
            unit=result['unit'],
            insight=result['insight'],
            timestamp=result['timestamp']
        )

        return jsonify({'status': 'success', 'data': result})

    except ValueError as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/predict-compare', methods=['POST'])
@login_required
def api_predict_compare():
    """
    Melakukan prediksi menggunakan KEDUA model sekaligus.
    Mengembalikan hasil perbandingan dan insight otomatis.
    """
    try:
        model = get_user_model()
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

        # Simpan ke database SQLite (riwayat best model)
        username = session['username']
        best_model_type = result['best_model']
        best_model_name = f"{best_model_type.replace('_', ' ').title()} (Best)"
        best_pred = result[best_model_type]['prediction']
        insight_text = '\n'.join(result['insight']) if isinstance(result['insight'], list) else str(result['insight'])

        save_prediction(
            username=username,
            model_name=best_model_name,
            model_type=best_model_type,
            input_data=input_data,
            prediction_value=best_pred,
            unit='% (Pertumbuhan GDP)',
            insight=insight_text,
            timestamp=result['timestamp']
        )

        return jsonify({'status': 'success', 'data': result})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/history', methods=['GET'])
@login_required
def api_history():
    """
    Mengembalikan riwayat prediksi milik user dari SQLite.
    """
    try:
        username = session['username']
        history = get_user_history(username)
        return jsonify({
            'status': 'success',
            'data': history,
            'total': len(history)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/history/<int:pred_id>', methods=['DELETE'])
@login_required
def api_delete_history_item(pred_id):
    """
    Melakukan soft-delete (menyembunyikan) item riwayat dari sisi user.
    """
    try:
        username = session['username']
        success = soft_delete_prediction(username, pred_id)
        if success:
            return jsonify({'status': 'success', 'message': 'Item riwayat berhasil dihapus dari tampilan Anda'})
        else:
            return jsonify({'status': 'error', 'message': 'Item riwayat tidak ditemukan'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/upload-csv', methods=['POST'])
@login_required
def api_upload_csv():
    """
    Upload file CSV atau Excel baru untuk melatih ulang model.
    """
    try:
        model = get_user_model()
        sid = get_session_id()

        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'Tidak ada file yang diunggah'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'Nama file kosong'}), 400

        if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
            return jsonify({'status': 'error', 'message': 'Hanya file CSV atau Excel (.xlsx, .xls) yang diizinkan'}), 400

        # Ambil metadata tambahan dari request form
        description = request.form.get('description', '')
        source = request.form.get('source', '')
        uploaded_by = request.form.get('uploaded_by', '')

        # Simpan file yang diunggah ke folder session
        user_upload_dir = os.path.join(UPLOAD_DIR, sid)
        os.makedirs(user_upload_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"uploaded_{timestamp}_{file.filename}"
        file_path = os.path.join(user_upload_dir, filename)
        file.save(file_path)

        # Latih model dengan dataset baru
        metrics = model.load_and_train(file_path)

        # Tulis metadata ke file JSON berdampingan dengan dataset
        metadata_path = file_path + '.json'
        metadata_content = {
            'original_filename': file.filename,
            'uploaded_by': uploaded_by if uploaded_by.strip() else 'Anonymous',
            'description': description if description.strip() else 'Tidak ada deskripsi',
            'source': source if source.strip() else 'Tidak ada sumber',
            'uploaded_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'rows': model.df.shape[0] if model.df is not None else 0,
            'cols': model.df.shape[1] if model.df is not None else 0
        }
        with open(metadata_path, 'w', encoding='utf-8') as f_meta:
            json.dump(metadata_content, f_meta, indent=4, ensure_ascii=False)

        # Simpan status dataset aktif ke database
        username = session['username']
        set_user_active_dataset(username, file_path)

        return jsonify({
            'status': 'success',
            'message': 'Dataset berhasil diunggah dan model telah dilatih',
            'filename': filename,
            'metrics': metrics
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/configure-model', methods=['POST'])
@login_required
def api_configure_model():
    """
    Mengonfigurasi ulang model dengan memilih fitur dan target baru,
    kemudian melatih ulang model.
    """
    try:
        model = get_user_model()
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
@login_required
def api_export():
    """
    Export riwayat prediksi dalam format CSV atau Excel.
    """
    try:
        username = session['username']
        format_type = request.args.get('format', 'csv')
        history = get_user_history(username)

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
@login_required
def api_infographic_data():
    """
    Mengembalikan data lengkap untuk halaman Infografis Otomatis.
    Menggabungkan dashboard, EDA, model comparison, dan trend menjadi satu payload.
    """
    try:
        model = get_user_model()
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
                'xgb_metrics': comparison.get('xgboost', {}).get('metrics', {}),
                'xgb_available': comparison.get('xgboost', {}).get('available', False),
                'summary': comparison['summary'],
            }
            # Include feature importance from RF
            if 'feature_importance' in comparison.get('random_forest', {}):
                comparison_summary['feature_importance'] = comparison['random_forest']['feature_importance']
            if 'feature_importance' in comparison.get('xgboost', {}):
                comparison_summary['xgb_feature_importance'] = comparison['xgboost']['feature_importance']

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
@login_required
def api_clear_history():
    """Menghapus semua riwayat prediksi."""
    try:
        model = get_user_model()
        model.clear_history()
        # Also clear predictions for user in SQLite
        username = session['username']
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('UPDATE predictions SET deleted_by_user = 1 WHERE username = ?', (username,))
        conn.commit()
        conn.close()
        return jsonify({'status': 'success', 'message': 'Riwayat prediksi berhasil dihapus'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ===========================================================
#                    USER DATASETS & ACTIONS
# ===========================================================

@app.route('/api/user/datasets', methods=['GET'])
@login_required
def api_user_datasets():
    try:
        username = session['username']
        user_upload_dir = os.path.join(UPLOAD_DIR, username)
        datasets = []
        if os.path.exists(user_upload_dir):
            for f in os.listdir(user_upload_dir):
                if f.endswith(('.csv', '.xlsx', '.xls')):
                    file_path = os.path.join(user_upload_dir, f)
                    stats = os.stat(file_path)
                    meta_path = file_path + '.json'
                    metadata = {}
                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path, 'r', encoding='utf-8') as mf:
                                metadata = json.load(mf)
                        except:
                            pass
                    rows, cols = 0, 0
                    try:
                        import pandas as pd
                        if f.lower().endswith(('.xlsx', '.xls')):
                            df_full = pd.read_excel(file_path)
                        else:
                            df_full = pd.read_csv(file_path)
                        rows, cols = df_full.shape
                    except:
                        pass
                    model = get_user_model()
                    is_active = (model.dataset_path == file_path)
                    datasets.append({
                        'filename': f,
                        'display_name': metadata.get('original_filename', f),
                        'size_kb': round(stats.st_size / 1024, 1),
                        'rows': rows,
                        'cols': cols,
                        'modified_at': datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        'is_active': is_active,
                        'source_type': 'user',
                        'session_id': username,
                        'metadata': metadata
                    })
        for f in ['ekonomi_data.csv', 'Data Wendy.csv']:
            file_path = os.path.join(DATASET_DIR, f)
            if os.path.exists(file_path):
                stats = os.stat(file_path)
                model = get_user_model()
                is_active = (model.dataset_path == file_path)
                datasets.append({
                    'filename': f,
                    'display_name': f,
                    'size_kb': round(stats.st_size / 1024, 1),
                    'rows': 40,
                    'cols': 10,
                    'modified_at': '-',
                    'is_active': is_active,
                    'source_type': 'system',
                    'session_id': '',
                    'metadata': {
                        'description': 'Dataset default bawaan sistem.',
                        'source': 'Sistem',
                        'uploaded_by': 'System',
                        'uploaded_at': '-'
                    }
                })
        return jsonify({'status': 'success', 'data': datasets})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/user/datasets/activate', methods=['POST'])
@login_required
def api_user_datasets_activate():
    try:
        username = session['username']
        model = get_user_model()
        data = request.get_json()
        if not data or 'filename' not in data or 'source_type' not in data:
            return jsonify({'status': 'error', 'message': 'Parameter tidak lengkap'}), 400
        filename = data['filename']
        source_type = data['source_type']
        if source_type == 'system':
            file_path = os.path.join(DATASET_DIR, filename)
        else:
            file_path = os.path.join(UPLOAD_DIR, username, filename)
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': 'Berkas dataset tidak ditemukan'}), 404
        metrics = model.load_and_train(file_path)
        set_user_active_dataset(username, file_path)
        return jsonify({
            'status': 'success',
            'message': f'Dataset {filename} berhasil diaktifkan',
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/user/datasets', methods=['DELETE'])
@login_required
def api_user_datasets_delete():
    try:
        username = session['username']
        filename = request.args.get('filename')
        if not filename:
            return jsonify({'status': 'error', 'message': 'Parameter filename harus diisi'}), 400
        file_path = os.path.join(UPLOAD_DIR, username, filename)
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': 'Berkas tidak ditemukan'}), 404
        model = get_user_model()
        reverted = False
        if model.dataset_path == file_path:
            model.load_and_train(DEFAULT_DATASET)
            set_user_active_dataset(username, None)
            reverted = True
        os.remove(file_path)
        meta_path = file_path + '.json'
        if os.path.exists(meta_path):
            os.remove(meta_path)
        return jsonify({
            'status': 'success',
            'message': f'Dataset {filename} berhasil dihapus',
            'reverted_to_default': reverted
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/user/datasets/preview', methods=['GET'])
@login_required
def api_user_datasets_preview():
    try:
        username = session['username']
        filename = request.args.get('filename')
        source_type = request.args.get('source_type', 'user')
        if not filename:
            return jsonify({'status': 'error', 'message': 'Parameter filename harus diisi'}), 400
        if source_type == 'system':
            file_path = os.path.join(DATASET_DIR, filename)
        else:
            file_path = os.path.join(UPLOAD_DIR, username, filename)
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': 'Berkas tidak ditemukan'}), 404
        import pandas as pd
        if filename.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path, nrows=10)
        else:
            df = pd.read_csv(file_path, nrows=10)
        df = df.where(pd.notnull(df), None)
        return jsonify({
            'status': 'success',
            'headers': list(df.columns),
            'rows': df.values.tolist()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/user/datasets/download', methods=['GET'])
@login_required
def api_user_datasets_download():
    try:
        username = session['username']
        filename = request.args.get('filename')
        source_type = request.args.get('source_type', 'user')
        if not filename:
            return jsonify({'status': 'error', 'message': 'Parameter filename harus diisi'}), 400
        if source_type == 'system':
            file_path = os.path.join(DATASET_DIR, filename)
        else:
            file_path = os.path.join(UPLOAD_DIR, username, filename)
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': 'Berkas tidak ditemukan'}), 404
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ===========================================================
#                    ADMIN API ENDPOINTS
# ===========================================================

@app.route('/api/admin/datasets', methods=['GET'])
@admin_required
def api_admin_datasets():
    """
    Daftar semua dataset yang tersedia (sistem + seluruh user uploads)
    beserta metadatanya.
    """
    try:
        current_model = get_user_model()
        datasets = []

        # 1. Scan folder dataset/ (sistem)
        if os.path.exists(DATASET_DIR):
            for f in os.listdir(DATASET_DIR):
                if f.endswith(('.csv', '.xlsx', '.xls')):
                    path = os.path.join(DATASET_DIR, f)
                    stats = os.stat(path)
                    
                    # Coba baca dimensi
                    try:
                        import pandas as pd
                        if f.endswith(('.xlsx', '.xls')):
                            df_temp = pd.read_excel(path)
                        else:
                            df_temp = pd.read_csv(path, sep=None, engine='python')
                        rows, cols = df_temp.shape
                    except Exception:
                        rows, cols = 0, 0
                        
                    datasets.append({
                        'filename': f,
                        'display_name': f,
                        'source_type': 'system',
                        'size_kb': round(stats.st_size / 1024, 2),
                        'modified_at': datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        'is_active': current_model.dataset_path == path,
                        'rows': rows,
                        'cols': cols,
                        'metadata': {
                            'original_filename': f,
                            'uploaded_by': 'System',
                            'description': 'Dataset default bawaan sistem.',
                            'source': 'Sistem',
                            'uploaded_at': '-'
                        }
                    })

        # 2. Scan folder uploads/ secara rekursif (pengguna)
        if os.path.exists(UPLOAD_DIR):
            import pandas as pd
            for root, dirs, files in os.walk(UPLOAD_DIR):
                for f in files:
                    if f.endswith(('.csv', '.xlsx', '.xls')):
                        path = os.path.join(root, f)
                        stats = os.stat(path)
                        
                        # Ambil session_id (nama subfolder langsung di bawah uploads/)
                        rel_path = os.path.relpath(root, UPLOAD_DIR)
                        session_id = rel_path.split(os.sep)[0] if rel_path and rel_path != '.' else 'unknown'
                        user_label = f"User ({session_id[:8]}...)" if session_id != 'unknown' else "User (Unknown)"

                        display_name = f
                        if f.startswith('uploaded_'):
                            parts = f.split('_', 2)
                            if len(parts) >= 3:
                                display_name = parts[2]

                        # Memuat metadata dari file .json jika ada
                        metadata = {
                            'original_filename': display_name,
                            'uploaded_by': user_label,
                            'description': 'Tidak ada deskripsi',
                            'source': 'Tidak diketahui',
                            'uploaded_at': datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                        }
                        meta_path = path + '.json'
                        if os.path.exists(meta_path):
                            try:
                                with open(meta_path, 'r', encoding='utf-8') as f_meta:
                                    metadata = json.load(f_meta)
                            except Exception:
                                pass

                        # Coba baca dimensi
                        try:
                            if f.endswith(('.xlsx', '.xls')):
                                df_temp = pd.read_excel(path)
                            else:
                                df_temp = pd.read_csv(path, sep=None, engine='python')
                            rows, cols = df_temp.shape
                        except Exception:
                            rows, cols = 0, 0

                        datasets.append({
                            'filename': f,
                            'display_name': display_name,
                            'source_type': 'user',
                            'user_label': user_label,
                            'session_id': session_id,
                            'size_kb': round(stats.st_size / 1024, 2),
                            'modified_at': datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                            'is_active': current_model.dataset_path == path,
                            'rows': rows,
                            'cols': cols,
                            'metadata': metadata
                        })

        return jsonify({'status': 'success', 'data': datasets})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/admin/datasets/activate', methods=['POST'])
@admin_required
def api_admin_datasets_activate():
    """
    Mengaktifkan dataset tertentu untuk sesi admin/user saat ini.
    """
    try:
        model = get_user_model()
        data = request.get_json()
        if not data or 'filename' not in data or 'source_type' not in data:
            return jsonify({'status': 'error', 'message': 'Parameter tidak lengkap'}), 400

        filename = data['filename']
        source_type = data['source_type']
        session_id = data.get('session_id', '')

        if source_type == 'system':
            file_path = os.path.join(DATASET_DIR, filename)
        else:
            if not session_id:
                session_id = get_session_id()
            file_path = os.path.join(UPLOAD_DIR, session_id, filename)

        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': 'Berkas dataset tidak ditemukan'}), 404

        # Latih model dengan dataset terpilih
        metrics = model.load_and_train(file_path)

        return jsonify({
            'status': 'success',
            'message': f'Dataset {filename} berhasil diaktifkan',
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/admin/datasets', methods=['DELETE'])
@admin_required
def api_admin_datasets_delete():
    """
    Menghapus berkas dataset milik pengguna.
    """
    try:
        filename = request.args.get('filename')
        session_id = request.args.get('session_id')
        source_type = request.args.get('source_type', 'user')

        if not filename or not session_id:
            return jsonify({'status': 'error', 'message': 'Parameter filename dan session_id harus diisi'}), 400

        if source_type == 'system':
            return jsonify({'status': 'error', 'message': 'Dataset sistem bawaan tidak boleh dihapus'}), 403

        file_path = os.path.join(UPLOAD_DIR, session_id, filename)
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': 'Berkas tidak ditemukan'}), 404

        # Reset model user mana pun yang sedang menggunakan berkas ini ke ekonomi_data.csv (default)
        reverted = False
        for sid, u_model in list(user_models.items()):
            if u_model.dataset_path == file_path:
                u_model.load_and_train(DEFAULT_DATASET)
                if sid == get_session_id():
                    reverted = True

        # Hapus berkas CSV/Excel dan berkas JSON metadatanya
        os.remove(file_path)
        meta_path = file_path + '.json'
        if os.path.exists(meta_path):
            os.remove(meta_path)

        # Hapus references active_dataset di users table database
        from database import get_db
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET active_dataset = NULL WHERE active_dataset = ?', (file_path,))
        conn.commit()
        conn.close()

        return jsonify({
            'status': 'success',
            'message': f'Dataset {filename} berhasil dihapus',
            'reverted_to_default': reverted
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/admin/datasets/download', methods=['GET'])
@admin_required
def api_admin_datasets_download():
    """
    Mengunduh berkas dataset terpilih.
    """
    try:
        filename = request.args.get('filename')
        session_id = request.args.get('session_id', '')
        source_type = request.args.get('source_type', 'user')

        if not filename:
            return jsonify({'status': 'error', 'message': 'Filename harus diisi'}), 400

        if source_type == 'system':
            file_path = os.path.join(DATASET_DIR, filename)
        else:
            if not session_id:
                session_id = get_session_id()
            file_path = os.path.join(UPLOAD_DIR, session_id, filename)

        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': 'Berkas tidak ditemukan'}), 404

        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/admin/datasets/preview', methods=['GET'])
@admin_required
def api_admin_datasets_preview():
    """
    Membaca dan mengembalikan 10 baris pertama dari dataset untuk preview.
    """
    try:
        filename = request.args.get('filename')
        session_id = request.args.get('session_id', '')
        source_type = request.args.get('source_type', 'user')

        if not filename:
            return jsonify({'status': 'error', 'message': 'Filename harus diisi'}), 400

        if source_type == 'system':
            file_path = os.path.join(DATASET_DIR, filename)
        else:
            if not session_id:
                session_id = get_session_id()
            file_path = os.path.join(UPLOAD_DIR, session_id, filename)

        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': 'Berkas tidak ditemukan'}), 404

        import pandas as pd
        if filename.lower().endswith(('.xlsx', '.xls')):
            df_temp = pd.read_excel(file_path, nrows=10)
        else:
            df_temp = pd.read_csv(file_path, sep=None, engine='python', nrows=10)

        # Ubah NaN menjadi None agar aman dibaca JSON
        df_temp = df_temp.where(pd.notnull(df_temp), None)

        headers = df_temp.columns.tolist()
        rows = df_temp.values.tolist()

        return jsonify({
            'status': 'success',
            'headers': headers,
            'rows': rows
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/admin/history', methods=['GET'])
@admin_required
def api_admin_history():
    """
    Mengembalikan seluruh riwayat prediksi semua pengguna dari SQLite (untuk audit admin).
    """
    try:
        history = get_all_history_admin()
        return jsonify({
            'status': 'success',
            'data': history,
            'total': len(history)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/admin/history/<int:pred_id>', methods=['DELETE'])
@admin_required
def api_admin_delete_history_item(pred_id):
    """
    Menghapus item riwayat prediksi secara permanen (hard-delete) dari SQLite.
    """
    try:
        success = hard_delete_prediction_admin(pred_id)
        if success:
            return jsonify({'status': 'success', 'message': 'Item riwayat berhasil dihapus secara permanen'})
        else:
            return jsonify({'status': 'error', 'message': 'Item riwayat tidak ditemukan'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ===========================================================
#                    MENJALANKAN SERVER
# ===========================================================

if __name__ == '__main__':
    print("\n>> Server berjalan di http://localhost:5000")
    print(">> Sistem Statistik Prediktif siap digunakan!\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
