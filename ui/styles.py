"""
Тёмная тема для BW Photo Studio.
Оптимизирована для работы с фотографиями (нейтральный серый фон).
"""

DARK_THEME = """
/* === Базовые виджеты === */
QWidget {
    background-color: #1e1e1e;
    color: #cccccc;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 10pt;
}

QMainWindow {
    background-color: #1a1a1a;
}

/* === Меню === */
QMenuBar {
    background-color: #2a2a2a;
    color: #cccccc;
    border-bottom: 1px solid #3a3a3a;
    padding: 2px;
}
QMenuBar::item { padding: 4px 10px; border-radius: 3px; }
QMenuBar::item:selected { background-color: #3d7ab5; color: #fff; }

QMenu {
    background-color: #2a2a2a;
    color: #cccccc;
    border: 1px solid #3a3a3a;
    padding: 4px 0;
}
QMenu::item { padding: 5px 24px 5px 10px; }
QMenu::item:selected { background-color: #3d7ab5; color: #fff; }
QMenu::separator { height: 1px; background: #3a3a3a; margin: 3px 8px; }

/* === Тулбар === */
QToolBar {
    background-color: #252525;
    border-bottom: 1px solid #333;
    spacing: 2px;
    padding: 4px 6px;
}
QToolBar::separator { width: 1px; background: #444; margin: 2px 4px; }

QToolButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 5px 7px;
    color: #cccccc;
    font-size: 9pt;
}
QToolButton:hover  { background: #3a3a3a; border-color: #555; }
QToolButton:pressed { background: #3d7ab5; }
QToolButton:checked { background: #2d5a8a; border-color: #3d7ab5; }

/* === Сплиттер === */
QSplitter::handle { background: #333; }
QSplitter::handle:horizontal { width: 2px; }
QSplitter::handle:vertical   { height: 2px; }
QSplitter::handle:hover { background: #555; }

/* === Вкладки === */
QTabWidget::pane {
    border: 1px solid #333;
    background: #222;
    border-radius: 0 3px 3px 3px;
}
QTabBar::tab {
    background: #2a2a2a;
    color: #999;
    padding: 8px 14px;
    border: 1px solid #333;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    min-width: 70px;
}
QTabBar::tab:selected   { background: #222; color: #fff; border-bottom: 2px solid #3d7ab5; }
QTabBar::tab:hover:!selected { background: #333; color: #ccc; }

/* === Скроллбары === */
QScrollBar:vertical {
    background: #1e1e1e; width: 10px; border: none; margin: 0;
}
QScrollBar::handle:vertical {
    background: #4a4a4a; border-radius: 5px; min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #5a5a5a; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QScrollBar:horizontal {
    background: #1e1e1e; height: 10px; border: none; margin: 0;
}
QScrollBar::handle:horizontal {
    background: #4a4a4a; border-radius: 5px; min-width: 24px;
}
QScrollBar::handle:horizontal:hover { background: #5a5a5a; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

/* === Кнопки === */
QPushButton {
    background: #333;
    color: #ccc;
    border: 1px solid #4a4a4a;
    border-radius: 4px;
    padding: 6px 16px;
    min-width: 72px;
}
QPushButton:hover  { background: #3a3a3a; border-color: #5a5a5a; color: #fff; }
QPushButton:pressed { background: #2a4a6a; border-color: #3d7ab5; }
QPushButton:disabled { background: #2a2a2a; color: #555; border-color: #333; }

QPushButton#primaryBtn {
    background: #3d7ab5; color: #fff; border-color: #4a8ac5;
    font-weight: bold;
}
QPushButton#primaryBtn:hover  { background: #4a8ac5; }
QPushButton#primaryBtn:pressed { background: #2a6aa5; }

QPushButton#resetBtn {
    background: #3a3a3a; color: #aaa;
}
QPushButton#resetBtn:hover { background: #444; color: #ccc; }

QPushButton#colorBtn {
    min-width: 60px; max-width: 60px; min-height: 24px; max-height: 24px;
    border-radius: 3px; border: 2px solid #666;
}
QPushButton#colorBtn:hover { border-color: #aaa; }

/* === Слайдеры === */
QSlider::groove:horizontal {
    background: #333; height: 4px; border-radius: 2px; border: none;
}
QSlider::handle:horizontal {
    background: #3d7ab5; border: none; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px;
}
QSlider::handle:horizontal:hover { background: #5090c5; }
QSlider::sub-page:horizontal { background: #3d7ab5; border-radius: 2px; }

/* === Метки === */
QLabel { color: #ccc; }
QLabel#sectionTitle {
    color: #fff; font-weight: bold; font-size: 11pt;
    padding: 6px 0 4px 0;
    border-bottom: 1px solid #3d7ab5;
}
QLabel#methodDesc {
    color: #888; font-size: 9pt;
    padding: 4px 6px;
    background: #252525;
    border-radius: 4px;
    border: 1px solid #333;
}
QLabel#infoLabel { color: #777; font-size: 9pt; }
QLabel#valueLabel { color: #5a9fd4; font-size: 9pt; min-width: 40px; }

/* === Группы === */
QGroupBox {
    border: 1px solid #3a3a3a;
    border-radius: 5px;
    margin-top: 10px;
    padding: 8px 6px 6px 6px;
    color: #888;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #999;
}

/* === Комбобоксы === */
QComboBox {
    background: #2e2e2e; color: #ccc;
    border: 1px solid #4a4a4a; border-radius: 4px;
    padding: 5px 8px; min-width: 120px;
}
QComboBox:hover { border-color: #5a5a5a; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox::down-arrow {
    image: none; border-left: 5px solid transparent;
    border-right: 5px solid transparent; border-top: 5px solid #888;
    margin-right: 6px;
}
QComboBox QAbstractItemView {
    background: #2a2a2a; color: #ccc; border: 1px solid #4a4a4a;
    selection-background-color: #3d7ab5;
    outline: none;
}

/* === Спинбоксы === */
QSpinBox, QDoubleSpinBox {
    background: #2e2e2e; color: #ccc;
    border: 1px solid #4a4a4a; border-radius: 4px; padding: 4px 6px;
}
QSpinBox:hover, QDoubleSpinBox:hover { border-color: #5a5a5a; }
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background: #333; border: none; width: 16px;
}

/* === Чекбоксы === */
QCheckBox { color: #ccc; spacing: 7px; }
QCheckBox::indicator {
    width: 15px; height: 15px;
    border: 1px solid #555; border-radius: 3px; background: #2e2e2e;
}
QCheckBox::indicator:checked {
    background: #3d7ab5; border-color: #3d7ab5;
}
QCheckBox::indicator:hover { border-color: #888; }

/* === Радиокнопки === */
QRadioButton { color: #ccc; spacing: 7px; }
QRadioButton::indicator {
    width: 15px; height: 15px;
    border: 1px solid #555; border-radius: 8px; background: #2e2e2e;
}
QRadioButton::indicator:checked { background: #3d7ab5; border-color: #3d7ab5; }
QRadioButton::indicator:hover { border-color: #888; }

/* === Список === */
QListWidget {
    background: #252525; border: 1px solid #333;
    border-radius: 4px; color: #ccc; outline: none;
}
QListWidget::item { padding: 7px 8px; border-bottom: 1px solid #2a2a2a; }
QListWidget::item:selected { background: #2d5a8a; color: #fff; }
QListWidget::item:hover:!selected { background: #2e2e2e; }

/* === Строка состояния === */
QStatusBar {
    background: #252525; color: #888;
    border-top: 1px solid #333; font-size: 9pt;
}
QStatusBar::item { border: none; }

/* === Прогресс-бар === */
QProgressBar {
    background: #2a2a2a; border: 1px solid #3a3a3a;
    border-radius: 4px; text-align: center; color: #ccc;
}
QProgressBar::chunk { background: #3d7ab5; border-radius: 3px; }

/* === Разделитель (горизонтальный) === */
QFrame[frameShape="4"] { color: #3a3a3a; }  /* HLine */
QFrame[frameShape="5"] { color: #3a3a3a; }  /* VLine */

/* === ScrollArea === */
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
"""
