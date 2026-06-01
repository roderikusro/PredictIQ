import sqlite3
import os
import json
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    # Create users table with active_dataset
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            active_dataset TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            model TEXT NOT NULL,
            model_type TEXT NOT NULL,
            input_data TEXT NOT NULL,
            prediction_value REAL NOT NULL,
            unit TEXT NOT NULL,
            insight TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            deleted_by_user INTEGER DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()
    print("[INFO] Database SQLite berhasil diinisialisasi.")

def register_user(username, password):
    if not username or not password:
        return False, "Username dan password tidak boleh kosong"
    
    username = username.strip().lower()
    if len(username) < 3:
        return False, "Username minimal 3 karakter"
    if len(password) < 6:
        return False, "Password minimal 6 karakter"
        
    password_hash = generate_password_hash(password)
    
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, password_hash))
        conn.commit()
        return True, "Registrasi berhasil"
    except sqlite3.IntegrityError:
        return False, "Username sudah terdaftar"
    finally:
        conn.close()

def login_user(username, password):
    username = username.strip().lower()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user and check_password_hash(user['password_hash'], password):
        return True, user['username']
    return False, "Username atau password salah"

def count_users():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM users')
    count = cursor.fetchone()[0]
    conn.close()
    return count

def get_user_active_dataset(username):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT active_dataset FROM users WHERE username = ?', (username,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row['active_dataset']
    return None

def set_user_active_dataset(username, dataset_path):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET active_dataset = ? WHERE username = ?', (dataset_path, username))
    conn.commit()
    conn.close()

def save_prediction(username, model_name, model_type, input_data, prediction_value, unit, insight, timestamp):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions 
        (username, model, model_type, input_data, prediction_value, unit, insight, timestamp, deleted_by_user)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
    ''', (username, model_name, model_type, json.dumps(input_data), prediction_value, unit, insight, timestamp))
    conn.commit()
    pred_id = cursor.lastrowid
    conn.close()
    return pred_id

def get_user_history(username):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM predictions 
        WHERE username = ? AND deleted_by_user = 0 
        ORDER BY timestamp ASC
    ''', (username,))
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for r in rows:
        history.append({
            'id': r['id'],
            'username': r['username'],
            'model': r['model'],
            'model_type': r['model_type'],
            'input': json.loads(r['input_data']),
            'prediction': r['prediction_value'],
            'unit': r['unit'],
            'insight': r['insight'],
            'timestamp': r['timestamp']
        })
    return history

def get_all_history_admin():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM predictions 
        ORDER BY timestamp ASC
    ''')
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for r in rows:
        history.append({
            'id': r['id'],
            'username': r['username'],
            'model': r['model'],
            'model_type': r['model_type'],
            'input': json.loads(r['input_data']),
            'prediction': r['prediction_value'],
            'unit': r['unit'],
            'insight': r['insight'],
            'timestamp': r['timestamp'],
            'deleted_by_user': r['deleted_by_user']
        })
    return history

def soft_delete_prediction(username, pred_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE predictions 
        SET deleted_by_user = 1 
        WHERE id = ? AND username = ?
    ''', (pred_id, username))
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0

def hard_delete_prediction_admin(pred_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM predictions WHERE id = ?', (pred_id,))
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0

def clear_all_history():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM predictions')
    conn.commit()
    conn.close()
