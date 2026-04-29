@echo off
setlocal

set "ROOT=%~dp0"
set "BACKEND_DIR=%ROOT%backend"
set "FRONTEND_DIR=%ROOT%frontend"

if not exist "%BACKEND_DIR%\.venv\Scripts\python.exe" (
  echo [SkinSight] Creating backend virtual environment...
  py -3.12 -m venv "%BACKEND_DIR%\.venv" 2>nul || python -m venv "%BACKEND_DIR%\.venv"
)

echo [SkinSight] Installing backend dependencies...
call "%BACKEND_DIR%\.venv\Scripts\python.exe" -m pip install -r "%BACKEND_DIR%\requirements.txt"
if errorlevel 1 (
  echo [SkinSight] Backend dependency install failed.
  pause
  exit /b 1
)

echo [SkinSight] Installing frontend dependencies...
pushd "%FRONTEND_DIR%"
call npm install
if errorlevel 1 (
  echo [SkinSight] Frontend dependency install failed.
  popd
  pause
  exit /b 1
)
popd

echo [SkinSight] Starting backend at http://127.0.0.1:8000
start "SkinSight Backend" cmd /k "cd /d "%BACKEND_DIR%" && .venv\Scripts\uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"

echo [SkinSight] Starting frontend at http://127.0.0.1:5173
start "SkinSight Frontend" cmd /k "cd /d "%FRONTEND_DIR%" && npm run dev -- --host 127.0.0.1 --port 5173"

timeout /t 3 >nul
start "" "http://127.0.0.1:5173"

echo [SkinSight] All services started.
echo [SkinSight] Browser opened at http://127.0.0.1:5173
endlocal
