@echo off
echo ============================================================
echo   BW Photo Studio - Установка зависимостей
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ОШИБКА] Python не найден!
    echo Загрузите Python 3.10+ с https://www.python.org/downloads/
    echo При установке отметьте "Add Python to PATH"
    pause
    exit /b 1
)

echo [OK] Python найден.
echo.
echo Установка зависимостей...
echo.

pip install --upgrade pip
pip install PyQt5>=5.15.9
pip install numpy>=1.23.0
pip install opencv-python>=4.7.0
pip install rawpy>=0.18.0
pip install Pillow>=10.0.0
pip install pillow-heif>=0.13.0
pip install scipy>=1.10.0
pip install imageio>=2.31.0

echo.
echo ============================================================
echo   Установка завершена!
echo   Запустите приложение командой: run.bat
echo ============================================================
pause
