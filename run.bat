@echo off
REM ============================================================
REM  Statisy — Quick Run Script
REM  Aktivasi venv, cek dependensi, jalankan Flask, buka browser
REM  Mendukung preview via jaringan (LAN share)
REM ============================================================
setlocal enabledelayedexpansion
pushd "%~dp0"
title Statisy — Predictive Analytics Server
color 0B

echo.
echo  =============================================
echo    Statisy - Sistem Statistik Prediktif
echo  =============================================
echo.

REM --- 1. Aktivasi Virtual Environment ---
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
    echo  [OK] Virtual environment aktif
) else (
    echo  [!!] Virtual environment tidak ditemukan.
    echo       Buat dulu dengan: python -m venv .venv
    echo.
    pause
    popd
    exit /b 1
)

REM --- 2. Cek Dependensi ---
echo.
echo  Memeriksa dependensi Python...
python -c "import flask; import sklearn; import xgboost" >nul 2>&1
if errorlevel 1 (
    echo  [..] Dependensi belum lengkap, menginstal...
    python -m pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo  [!!] Instalasi gagal.
        echo       Coba manual: pip install -r requirements.txt
        pause
        popd
        exit /b 1
    )
    echo  [OK] Semua dependensi terinstal
) else (
    echo  [OK] Semua dependensi sudah tersedia
)

REM --- 3. Deteksi IP Lokal ---
set "local_ip="
for /f "tokens=2 delims=:" %%A in ('ipconfig ^| findstr /R /C:"IPv4 Address" /C:"IPv4"') do (
    set "ip=%%A"
    set "ip=!ip: =!"
    if not "!ip!"=="127.0.0.1" if not "!ip!"=="0.0.0.0" (
        if not defined local_ip set "local_ip=!ip!"
    )
)

REM --- 4. Tampilkan Info Server ---
echo.
echo  =============================================
echo    Server berjalan di:
echo  ---------------------------------------------
echo    Lokal   :  http://localhost:5000
if defined local_ip (
    echo    Jaringan:  http://!local_ip!:5000
) else (
    echo    Jaringan:  ^(tidak terdeteksi^)
)
echo  =============================================
echo.
if defined local_ip (
    echo  Bagikan link ini untuk preview di perangkat lain
    echo  yang terhubung ke WiFi/LAN yang sama:
    echo.
    echo    http://!local_ip!:5000
    echo.
)
echo  Tekan Ctrl+C di jendela server untuk berhenti.
echo.

REM --- 5. Buka Browser Otomatis ---
timeout /t 2 /nobreak >nul
if defined local_ip (
    start "" "http://!local_ip!:5000"
) else (
    start "" "http://localhost:5000"
)

REM --- 6. Jalankan Flask (foreground, bukan window terpisah) ---
cd backend
python app.py

REM --- Cleanup saat server berhenti ---
echo.
echo  Server dihentikan.
popd
endlocal
pause
