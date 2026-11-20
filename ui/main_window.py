"""
Главное окно BW Photo Studio.
"""

import os
import sys
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QTabWidget, QLabel, QStatusBar, QAction, QFileDialog,
    QMessageBox, QProgressDialog, QApplication, QToolBar,
    QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QSettings, QTimer
from PyQt5.QtGui import QIcon, QKeySequence, QPixmap, QColor

from ui.image_viewer import ImageViewer, HistogramWidget, ndarray_to_qpixmap
from ui.panels import BWConversionPanel, SepiaPanel, EditingPanel, InfraredPanel
from core.image_io import (load_image, save_image, get_file_filter,
                            get_save_filter, get_image_info,
                            raw_available, heif_available)


APP_NAME    = 'BW Photo Studio'
APP_VERSION = '1.0'


# ---------------------------------------------------------------------------
# Поток загрузки изображения (чтобы UI не зависал при RAW)
# ---------------------------------------------------------------------------

class LoadThread(QThread):
    finished = pyqtSignal(np.ndarray, dict)
    error    = pyqtSignal(str)

    def __init__(self, filepath: str):
        super().__init__()
        self._path = filepath

    def run(self):
        try:
            image, meta = load_image(self._path)
            self.finished.emit(image, meta)
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Главное окно
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1400, 900)

        # --- Данные ---
        self._original: np.ndarray | None = None
        self._working:  np.ndarray | None = None   # Текущее состояние
        self._processed: np.ndarray | None = None  # Результат последней операции
        self._metadata: dict = {}

        self._undo_stack: list[np.ndarray] = []
        self._redo_stack: list[np.ndarray] = []
        self._max_undo = 30

        self._settings = QSettings(APP_NAME, APP_NAME)
        self._load_thread: LoadThread | None = None
        self._recent_files: list[str] = []
        self._load_recent_files()

        # --- Интерфейс ---
        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self._build_statusbar()
        self._connect_panels()
        self._restore_geometry()
        self._update_title()
        self._update_actions()

    # ==================================================================
    # Построение UI
    # ==================================================================

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Основной сплиттер ---
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        main_layout.addWidget(splitter)

        # --- Левая панель: вкладки инструментов ---
        self._tools_tab = QTabWidget()
        self._tools_tab.setMinimumWidth(240)
        self._tools_tab.setMaximumWidth(380)
        self._tools_tab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        self._bw_panel    = BWConversionPanel()
        self._sepia_panel = SepiaPanel()
        self._edit_panel  = EditingPanel()
        self._ir_panel    = InfraredPanel()

        self._tools_tab.addTab(self._bw_panel,    'Ч/Б')
        self._tools_tab.addTab(self._sepia_panel, 'Сепия')
        self._tools_tab.addTab(self._edit_panel,  'Правка')
        self._tools_tab.addTab(self._ir_panel,    'ИК')

        self._tools_tab.setTabToolTip(0, 'Конвертация в чёрно-белый')
        self._tools_tab.setTabToolTip(1, 'Тонирование в сепию')
        self._tools_tab.setTabToolTip(2, 'Редактирование Ч/Б фотографии')
        self._tools_tab.setTabToolTip(3, 'Инфракрасная фотография')

        splitter.addWidget(self._tools_tab)

        # --- Центральная панель: просмотр ---
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        self._viewer = ImageViewer()
        center_layout.addWidget(self._viewer)

        # Гистограмма внизу под просмотром
        self._histogram = HistogramWidget()
        center_layout.addWidget(self._histogram)

        splitter.addWidget(center_widget)

        splitter.setSizes([280, 1200])
        self._splitter = splitter

        # Соединяем сигналы viewer
        self._viewer.zoom_changed.connect(self._on_zoom_changed)
        self._viewer.cursor_moved.connect(self._on_cursor_moved)

    def _build_menu(self):
        mb = self.menuBar()

        # --- Файл ---
        file_menu = mb.addMenu('Файл')

        self._act_open = QAction('Открыть…', self, shortcut='Ctrl+O')
        self._act_open.setStatusTip('Открыть изображение (JPEG, PNG, TIFF, NEF, NRW и другие форматы)')
        self._act_open.triggered.connect(self.open_file)
        file_menu.addAction(self._act_open)

        self._recent_menu = file_menu.addMenu('Последние файлы')
        self._rebuild_recent_menu()

        file_menu.addSeparator()

        self._act_save = QAction('Сохранить как…', self, shortcut='Ctrl+Shift+S')
        self._act_save.setEnabled(False)
        self._act_save.triggered.connect(self.save_file)
        file_menu.addAction(self._act_save)

        self._act_export_orig = QAction('Экспорт оригинала…', self)
        self._act_export_orig.setEnabled(False)
        self._act_export_orig.triggered.connect(self.export_original)
        file_menu.addAction(self._act_export_orig)

        file_menu.addSeparator()

        act_exit = QAction('Выход', self, shortcut='Alt+F4')
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # --- Правка ---
        edit_menu = mb.addMenu('Правка')

        self._act_undo = QAction('Отменить', self, shortcut='Ctrl+Z')
        self._act_undo.setEnabled(False)
        self._act_undo.triggered.connect(self.undo)
        edit_menu.addAction(self._act_undo)

        self._act_redo = QAction('Повторить', self, shortcut='Ctrl+Y')
        self._act_redo.setEnabled(False)
        self._act_redo.triggered.connect(self.redo)
        edit_menu.addAction(self._act_redo)

        edit_menu.addSeparator()

        self._act_reset = QAction('Сбросить все изменения', self, shortcut='Ctrl+R')
        self._act_reset.setEnabled(False)
        self._act_reset.triggered.connect(self.reset_to_original)
        edit_menu.addAction(self._act_reset)

        # --- Вид ---
        view_menu = mb.addMenu('Вид')

        self._act_fit = QAction('Вписать в окно', self, shortcut='Ctrl+0')
        self._act_fit.triggered.connect(self._viewer.fit_view)
        view_menu.addAction(self._act_fit)

        self._act_actual = QAction('100% (оригинальный размер)', self, shortcut='Ctrl+1')
        self._act_actual.triggered.connect(self._viewer.zoom_actual)
        view_menu.addAction(self._act_actual)

        self._act_zin = QAction('Увеличить', self, shortcut='Ctrl+=')
        self._act_zin.triggered.connect(self._viewer.zoom_in)
        view_menu.addAction(self._act_zin)

        self._act_zout = QAction('Уменьшить', self, shortcut='Ctrl+-')
        self._act_zout.triggered.connect(self._viewer.zoom_out)
        view_menu.addAction(self._act_zout)

        view_menu.addSeparator()

        self._act_split = QAction('Режим сравнения (ДО/ПОСЛЕ)', self,
                                  shortcut='Ctrl+D', checkable=True, checked=True)
        self._act_split.triggered.connect(lambda c: self._viewer.set_split_mode(c))
        view_menu.addAction(self._act_split)

        self._act_histo = QAction('Показать гистограмму', self,
                                   checkable=True, checked=True)
        self._act_histo.triggered.connect(lambda c: self._histogram.setVisible(c))
        view_menu.addAction(self._act_histo)

        # --- Справка ---
        help_menu = mb.addMenu('Справка')

        act_about = QAction('О программе…', self)
        act_about.triggered.connect(self.show_about)
        help_menu.addAction(act_about)

        act_formats = QAction('Поддерживаемые форматы', self)
        act_formats.triggered.connect(self.show_formats)
        help_menu.addAction(act_formats)

    def _build_toolbar(self):
        tb = QToolBar('Основная панель')
        tb.setObjectName('mainToolbar')
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)

        tb.addAction(self._act_open)
        self._act_save_tb = tb.addAction('Сохранить')
        self._act_save_tb.setEnabled(False)
        self._act_save_tb.triggered.connect(self.save_file)

        tb.addSeparator()
        tb.addAction(self._act_undo)
        tb.addAction(self._act_redo)

        tb.addSeparator()
        tb.addAction(self._act_fit)
        tb.addAction(self._act_actual)

        tb.addSeparator()
        tb.addAction(self._act_split)

        tb.addSeparator()
        self._act_reset_tb = tb.addAction('Сбросить')
        self._act_reset_tb.setEnabled(False)
        self._act_reset_tb.triggered.connect(self.reset_to_original)

    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)

        self._sb_file  = QLabel('Нет изображения')
        self._sb_size  = QLabel('')
        self._sb_zoom  = QLabel('')
        self._sb_pixel = QLabel('')
        self._sb_raw   = QLabel('')

        for lbl in [self._sb_file, self._sb_size, self._sb_zoom,
                    self._sb_pixel, self._sb_raw]:
            lbl.setStyleSheet('color: #888; padding: 0 8px;')
            sb.addWidget(lbl)

        # Индикаторы поддержки форматов справа
        raw_ok = raw_available()
        heif_ok = heif_available()
        caps = []
        caps.append(f'RAW: {"✓" if raw_ok else "✗"}')
        caps.append(f'HEIF: {"✓" if heif_ok else "✗"}')
        cap_lbl = QLabel('  '.join(caps))
        cap_lbl.setStyleSheet(
            'color: #3d7ab5; padding: 0 8px; font-size: 9pt;'
        )
        sb.addPermanentWidget(cap_lbl)

    def _connect_panels(self):
        for panel in [self._bw_panel, self._sepia_panel,
                      self._edit_panel, self._ir_panel]:
            panel.result_ready.connect(self._on_panel_result)

    # ==================================================================
    # Файловые операции
    # ==================================================================

    def open_file(self):
        last_dir = self._settings.value('last_open_dir', os.path.expanduser('~'))
        filepath, _ = QFileDialog.getOpenFileName(
            self, 'Открыть изображение', last_dir, get_file_filter()
        )
        if not filepath:
            return
        self._settings.setValue('last_open_dir', os.path.dirname(filepath))
        self._do_load(filepath)

    def _do_load(self, filepath: str):
        self._sb_file.setText(f'Загрузка: {os.path.basename(filepath)}…')
        QApplication.processEvents()

        progress = QProgressDialog('Загрузка файла…', None, 0, 0, self)
        progress.setWindowTitle('Пожалуйста, подождите')
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(300)
        progress.show()

        self._load_thread = LoadThread(filepath)
        self._load_thread.finished.connect(lambda img, meta: self._on_load_done(img, meta, progress))
        self._load_thread.error.connect(lambda e: self._on_load_error(e, progress))
        self._load_thread.start()

    def _on_load_done(self, image: np.ndarray, meta: dict, progress):
        progress.close()

        self._original = image.copy()
        self._working  = image.copy()
        self._processed = None
        self._metadata  = meta
        self._undo_stack.clear()
        self._redo_stack.clear()

        # Обновляем все панели
        for panel in [self._bw_panel, self._sepia_panel,
                      self._edit_panel, self._ir_panel]:
            panel.set_source(self._working)

        self._viewer.set_original(self._original)
        self._viewer.set_processed(None)
        self._histogram.set_image(self._original)

        QTimer.singleShot(100, self._viewer.fit_view)

        # Обновляем статус
        info = get_image_info(image, meta)
        self._sb_file.setText(meta.get('filename', ''))
        self._sb_size.setText(info)
        self._update_title(meta.get('filename', ''))
        self._add_recent(meta['path'])
        self._update_actions()

    def _on_load_error(self, error_msg: str, progress):
        progress.close()
        QMessageBox.critical(
            self, 'Ошибка загрузки',
            f'Не удалось открыть файл:\n\n{error_msg}\n\n'
            'Убедитесь, что установлены все зависимости (install.bat).'
        )
        self._sb_file.setText('Ошибка загрузки')

    def save_file(self):
        if self._working is None:
            return
        last_dir = self._settings.value('last_save_dir', os.path.expanduser('~'))
        orig_name = os.path.splitext(self._metadata.get('filename', 'image'))[0]
        default_path = os.path.join(last_dir, orig_name + '_bw.jpg')

        filepath, flt = QFileDialog.getSaveFileName(
            self, 'Сохранить изображение', default_path, get_save_filter()
        )
        if not filepath:
            return

        # Добавляем расширение если нет
        ext_map = {'JPEG файлы': '.jpg', 'PNG файлы': '.png',
                   'TIFF файлы': '.tiff', 'BMP файлы': '.bmp'}
        for key, ext in ext_map.items():
            if key in flt and not filepath.lower().endswith(ext):
                filepath += ext
                break

        self._settings.setValue('last_save_dir', os.path.dirname(filepath))

        img_to_save = self._processed if self._processed is not None else self._working
        ok = save_image(img_to_save, filepath)

        if ok:
            self._sb_file.setText(f'Сохранено: {os.path.basename(filepath)}')
        else:
            QMessageBox.critical(self, 'Ошибка', f'Не удалось сохранить файл:\n{filepath}')

    def export_original(self):
        """Сохранить оригинальное (конвертированное из RAW) изображение."""
        if self._original is None:
            return
        last_dir = self._settings.value('last_save_dir', os.path.expanduser('~'))
        orig_name = os.path.splitext(self._metadata.get('filename', 'image'))[0]
        default_path = os.path.join(last_dir, orig_name + '_original.tiff')

        filepath, _ = QFileDialog.getSaveFileName(
            self, 'Экспорт оригинала', default_path,
            'TIFF файлы (*.tiff);;PNG файлы (*.png);;JPEG файлы (*.jpg)'
        )
        if filepath:
            save_image(self._original, filepath, quality=100)

    # ==================================================================
    # Отмена / повтор
    # ==================================================================

    def _push_undo(self, state: np.ndarray):
        self._undo_stack.append(state.copy())
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._update_actions()

    def undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(self._working.copy())
        self._working = self._undo_stack.pop()
        self._processed = None
        self._update_view()
        self._update_actions()

    def redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(self._working.copy())
        self._working = self._redo_stack.pop()
        self._processed = None
        self._update_view()
        self._update_actions()

    def reset_to_original(self):
        if self._original is None:
            return
        reply = QMessageBox.question(
            self, 'Подтверждение',
            'Сбросить все изменения и вернуться к оригиналу?',
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._push_undo(self._working)
            self._working = self._original.copy()
            self._processed = None
            self._update_view()

    # ==================================================================
    # Результат панели
    # ==================================================================

    def _on_panel_result(self, result: np.ndarray):
        """Получаем обработанное изображение от панели."""
        self._processed = result

        # Обновляем просмотр
        self._viewer.set_original(self._working)
        self._viewer.set_processed(result)
        self._histogram.set_image(result)

        # Обновляем действия
        self._update_actions()

        # Автоматически принимаем результат в рабочий слой,
        # но оставляем возможность сравнения через viewer
        self._push_undo(self._working)
        self._working = result.copy()

        # Обновляем источник для всех панелей
        for panel in [self._bw_panel, self._sepia_panel,
                      self._edit_panel, self._ir_panel]:
            panel.set_source(self._working)

    def _update_view(self):
        """Обновляет просмотр при undo/redo."""
        self._viewer.set_original(self._original)
        self._viewer.set_processed(self._working if self._working is not self._original else None)
        self._histogram.set_image(self._working)
        for panel in [self._bw_panel, self._sepia_panel,
                      self._edit_panel, self._ir_panel]:
            panel.set_source(self._working)

    # ==================================================================
    # Статусная строка
    # ==================================================================

    def _on_zoom_changed(self, zoom: float):
        self._sb_zoom.setText(f'{zoom * 100:.0f}%')

    def _on_cursor_moved(self, x: int, y: int):
        if self._working is None:
            return
        h, w = self._working.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            px = self._working[y, x]
            if len(px) >= 3:
                b, g, r = px[0], px[1], px[2]
                lum = int(0.2126 * r + 0.7152 * g + 0.0722 * b)
                self._sb_pixel.setText(f'x={x} y={y}  R={r} G={g} B={b}  L={lum}')

    # ==================================================================
    # Недавние файлы
    # ==================================================================

    def _load_recent_files(self):
        self._recent_files = self._settings.value('recent_files', []) or []
        if isinstance(self._recent_files, str):
            self._recent_files = [self._recent_files]

    def _add_recent(self, filepath: str):
        if filepath in self._recent_files:
            self._recent_files.remove(filepath)
        self._recent_files.insert(0, filepath)
        self._recent_files = self._recent_files[:10]
        self._settings.setValue('recent_files', self._recent_files)
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self):
        self._recent_menu.clear()
        for f in self._recent_files:
            if os.path.exists(f):
                act = QAction(os.path.basename(f), self)
                act.setStatusTip(f)
                act.setData(f)
                act.triggered.connect(lambda checked, fp=f: self._do_load(fp))
                self._recent_menu.addAction(act)
        if not self._recent_files:
            act = QAction('(нет)', self)
            act.setEnabled(False)
            self._recent_menu.addAction(act)
        else:
            self._recent_menu.addSeparator()
            clear_act = QAction('Очистить список', self)
            clear_act.triggered.connect(self._clear_recent)
            self._recent_menu.addAction(clear_act)

    def _clear_recent(self):
        self._recent_files.clear()
        self._settings.setValue('recent_files', [])
        self._rebuild_recent_menu()

    # ==================================================================
    # Вспомогательные методы
    # ==================================================================

    def _update_title(self, filename: str = ''):
        if filename:
            self.setWindowTitle(f'{filename} — {APP_NAME}')
        else:
            self.setWindowTitle(APP_NAME)

    def _update_actions(self):
        has_image = self._working is not None
        self._act_save.setEnabled(has_image)
        self._act_save_tb.setEnabled(has_image)
        self._act_export_orig.setEnabled(has_image)
        self._act_reset.setEnabled(has_image)
        self._act_reset_tb.setEnabled(has_image)
        self._act_undo.setEnabled(len(self._undo_stack) > 0)
        self._act_redo.setEnabled(len(self._redo_stack) > 0)
        undo_count = len(self._undo_stack)
        self._act_undo.setText(f'Отменить ({undo_count})' if undo_count else 'Отменить')

    def _restore_geometry(self):
        geom = self._settings.value('geometry')
        if geom:
            self.restoreGeometry(geom)
        state = self._settings.value('windowState')
        if state:
            self.restoreState(state)

    def closeEvent(self, event):
        self._settings.setValue('geometry', self.saveGeometry())
        self._settings.setValue('windowState', self.saveState())
        event.accept()

    # ==================================================================
    # Диалоги
    # ==================================================================

    def show_about(self):
        raw_status = '✓ rawpy установлен' if raw_available() else '✗ rawpy не установлен'
        heif_status = '✓ pillow-heif установлен' if heif_available() else '✗ pillow-heif не установлен'

        QMessageBox.about(
            self, f'О программе {APP_NAME}',
            f'<h2>{APP_NAME} v{APP_VERSION}</h2>'
            f'<p>Редактор чёрно-белой и инфракрасной цифровой фотографии.</p>'
            f'<p><b>Поддержка форматов:</b><br>'
            f'{raw_status}<br>'
            f'{heif_status}</p>'
            f'<p><b>Поддерживаемые форматы файлов:</b><br>'
            f'JPEG, PNG, TIFF, BMP, WebP<br>'
            f'RAW: Nikon NEF/NRW, Canon CR2/CR3, Sony ARW, Fujifilm RAF,<br>'
            f'Olympus ORF, Panasonic RW2, Pentax PEF, Adobe DNG и другие<br>'
            f'HEIF/HEIC (iOS, Nikon Z6 III)</p>'
            f'<p>Работает полностью автономно — без интернета и ИИ.</p>'
        )

    def show_formats(self):
        from core.image_io import get_supported_extensions
        exts = get_supported_extensions()
        ext_text = '  '.join(f'*{e}' for e in exts)
        QMessageBox.information(
            self, 'Поддерживаемые форматы',
            f'Поддерживаемые расширения файлов:\n\n{ext_text}\n\n'
            'RAW-поддержка через библиотеку rawpy (LibRaw).\n'
            'HEIF-поддержка через pillow-heif.\n'
            'Для Nikon Z6 III: NEF (14 бит), NRW, HEIF, TIFF, JPEG — всё поддерживается.'
        )
