"""
WSGI Configuration untuk PythonAnywhere
File ini digunakan oleh PythonAnywhere untuk menjalankan Flask app.
"""

import sys
import os

# Path ke project di PythonAnywhere (UBAH 'yourusername' dengan username kamu)
project_path = '/home/yourusername/Projek-Akhir'
backend_path = os.path.join(project_path, 'backend')

if project_path not in sys.path:
    sys.path.insert(0, project_path)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Set working directory
os.chdir(backend_path)

# pyrefly: ignore [missing-import]
from app import app as application
