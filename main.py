"""
BW Photo Studio — Редактор чёрно-белой и инфракрасной фотографии
Запуск: python main.py
"""

import sys
import os

# Добавляем директорию проекта в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from ui.main_window import MainWindow
from ui.styles import DARK_THEME


def main():
    # Включаем DPI-масштабирование для 4K-мониторов
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName('BW Photo Studio')
    app.setOrganizationName('BW Photo Studio')
    app.setApplicationVersion('1.0')

    # Применяем тёмную тему
    app.setStyleSheet(DARK_THEME)

    # Шрифт
    font = QFont('Segoe UI', 10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    # Если файл передан как аргумент командной строки
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        if os.path.isfile(filepath):
            window._do_load(filepath)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
