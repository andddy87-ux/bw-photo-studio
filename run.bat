@echo off
cd /d "%~dp0"
python main.py
if errorlevel 1 (
    echo.
    echo [ОШИБКА] Не удалось запустить приложение.
    echo Убедитесь, что выполнена установка: install.bat
    pause
)
