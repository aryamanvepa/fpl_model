@echo off
REM Launches the FPL Bot dashboard (Streamlit) and a public ngrok tunnel.
REM Double-click this file, or run it from a terminal.
REM Each part opens in its own window -- close a window to stop that part;
REM closing THIS window (if run from one) does not stop them.

cd /d D:\fpl

echo Starting Streamlit dashboard on http://localhost:8501 ...
start "FPL Dashboard - Streamlit" cmd /k "python -m streamlit run fpl_bot\dashboard\Home.py --server.headless true --server.port 8501"

timeout /t 3 /nobreak >nul

echo Starting ngrok public tunnel ...
start "FPL Dashboard - ngrok tunnel" cmd /k "ngrok http 8501"

echo.
echo Two windows should now be open:
echo   1. Streamlit dashboard (local, http://localhost:8501)
echo   2. ngrok tunnel -- check that window for your public URL, e.g. https://xxxx.ngrok-free.app
echo      (the free tier gives you a NEW url every time this script runs)
echo.
echo Leave both windows open while you're using the dashboard. Close either one to stop it.
pause
