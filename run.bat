@echo off
title Cardia - Heart Attack Risk Predictor
chcp 65001 > nul
cd /d "%~dp0"

echo.
echo  ============================================================
echo   Cardia — Heart Attack Risk Predictor
echo   Habiba Mohamed Hassan
echo  ============================================================
echo.

echo  Installing/updating Python packages from requirements.txt...
python -m pip install -q -r requirements.txt

echo.
echo  Launching the web app at http://127.0.0.1:5000
echo  (press Ctrl+C in this window to stop the server)
echo.

python app.py
pause
