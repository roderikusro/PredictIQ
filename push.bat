@echo off
chcp 65001 >nul
echo ============================================
echo   Statisy - Push Update ke GitHub
echo ============================================
echo.

cd /d "%~dp0"

:: Tambahkan semua perubahan
git add .

:: Minta pesan commit
set /p pesan="Pesan commit: "
if "%pesan%"=="" set pesan=Update project

:: Commit dan push
git commit -m "%pesan%"
git push origin main

echo.
echo ============================================
echo   Push berhasil!
echo   Repo: https://github.com/roderikusro/Statisy
echo ============================================
echo.
echo Jangan lupa reload di PythonAnywhere setelah push!
echo.
pause
