@echo off
REM ============================================================
REM  Personal Dictation Assistant - One-Click Setup
REM ============================================================

echo.
echo  *** Personal Dictation Assistant Setup ***
echo.

REM Check Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not on your PATH.
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo.
echo [2/4] Detecting GPU...
python -c "import subprocess; result = subprocess.run(['nvidia-smi'], capture_output=True); exit(0 if result.returncode==0 else 1)" >nul 2>&1
if %errorlevel% equ 0 (
    echo   NVIDIA GPU detected! Installing PyTorch with CUDA 12.1 support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
) else (
    echo   No NVIDIA GPU detected. Installing CPU-only PyTorch...
    pip install torch torchvision torchaudio --quiet
)

echo.
echo [3/4] Installing application dependencies...
pip install -r requirements.txt --quiet

echo.
echo [4/4] Setup complete!
echo.
echo To launch the app, run:
echo   venv\Scripts\activate
echo   python app.py
echo.
echo Or simply double-click "run.bat" after setup.
echo.

REM Create a simple run script
echo @echo off > run.bat
echo call venv\Scripts\activate.bat >> run.bat
echo python app.py >> run.bat

pause
